module Main where

import Control.Exception (IOException, bracket, try)
import Control.Monad (foldM, unless, when)
import qualified Data.Text as Text
import qualified Torch as T
import qualified Torch.Optim as Optim
import qualified Torch.Serialize as Serialize
import System.Directory (doesFileExist, getTemporaryDirectory, removeFile)
import System.IO (hClose, openBinaryTempFile)
import NanoChat
import NanoChat.Checkpoint
import NanoChat.Tokenizer
import NanoChat.Training

assert :: String -> Bool -> IO ()
assert label ok = unless ok (ioError (userError label))

main :: IO ()
main = do
  let tokenizer = buildTokenizer (Text.pack "hello hello world")
      token = tokenToId tokenizer . Text.pack
  assert "Vocabulary includes unique words and four reserved tokens" (tokenizerSize tokenizer == 6)
  assert "Unknown words map to UNK" (token "missing" == token "[UNK]")
  assert "Special token roundtrip" (idToToken tokenizer (token "[END]") == Text.pack "[END]")
  assert "Negative token IDs are safe" (idToToken tokenizer (-1) == Text.pack "[UNK]")
  let sequenceTokens = T.asTensor ([0..4] :: [Int])
  (bx, by) <- sampleBatch sequenceTokens 2 4
  assert "Minimal-length dataset and shifted targets"
    (T.asValue bx == ([[0,1,2,3],[0,1,2,3]] :: [[Int]]) &&
     T.asValue by == ([[1,2,3,4],[1,2,3,4]] :: [[Int]]))
  badBatch <- try (sampleBatch sequenceTokens 2 5) :: IO (Either IOException (T.Tensor, T.Tensor))
  assert "Reject too-short datasets" (either (const True) (const False) badBatch)
  let cfg = Config 4 8 2 1 6 0
  assert "Reject invalid head width" (either (const True) (const False) (validateConfig (cfg {nHead = 3})))
  model <- T.sample (NanoChatModelSpec cfg)
  let x = T.asTensor ([[0,1,2,3],[1,2,3,4]] :: [[Int]])
      y = T.asTensor ([[1,2,3,4],[2,3,4,5]] :: [[Int]])
      (logits, initialMaybe) = modelForward model cfg x (Just y)
  initialLoss <- maybe (ioError (userError "Missing initial loss")) pure initialMaybe
  let initial = T.asValue initialLoss :: Float
  assert "Logits have [batch,time,vocab] shape" (T.shape logits == [2,4,6])
  assert "Finite loss" (not (isNaN initial || isInfinite initial))
  let changed = T.asTensor ([[0,1,5,5],[1,2,0,0]] :: [[Int]])
      (changedLogits, _) = modelForward model cfg changed Nothing
      prefix a = T.sliceDim 1 0 2 1 a
      maxDiff = T.asValue (T.max (T.abs (prefix logits - prefix changedLogits))) :: Float
  assert "Future tokens cannot affect earlier logits" (maxDiff < 1e-5)
  let optimizer = Optim.mkAdam 0 0.9 0.999 (T.flattenParameters model)
  (trained, _) <- foldM (\(current, opt) _ -> do
      (updated, next, _) <- trainStep current cfg opt x y
      pure (updated, next)) (model, optimizer) [1..20 :: Int]
  let (_, finalMaybe) = modelForward trained cfg x (Just y)
  finalLoss <- maybe (ioError (userError "Missing final loss")) pure finalMaybe
  let final = T.asValue finalLoss :: Float
  assert "Adam reduces loss on a fixed batch" (final < initial)
  putStrLn ("Tiny training loss: " ++ show initial ++ " -> " ++ show final)
  generated <- generate trained cfg x 5
  let values = concat (T.asValue generated :: [[Int]])
  assert "Generation crosses context window" (T.shape generated == [2,9])
  assert "Generated IDs stay within vocabulary" (all (\i -> i >= 0 && i < vocabSize cfg) values)
  let evalCfg = cfg {dropout = 0.5}
      (evalA, _) = modelForward trained evalCfg x Nothing
      (evalB, _) = modelForward trained evalCfg x Nothing
  assert "Evaluation disables dropout"
    ((T.asValue (T.max (T.abs (evalA - evalB))) :: Float) == 0)
  (_, trainLoss) <- modelForwardTrain trained evalCfg x (Just y)
  case trainLoss of
    Nothing -> ioError (userError "Missing training loss")
    Just loss -> assert "Dropout training loss is finite"
      (let value = T.asValue loss :: Float in not (isNaN value || isInfinite value))
  -- Reload the trained model and its exact vocabulary without a dataset file.
  -- Non-ASCII text exercises UTF-8 metadata serialization as well.
  let savedTokenizer = buildTokenizer (Text.pack "hello grüß")
      info = ModelInfo 1 cfg (tokenizerVocabulary savedTokenizer) 20 "missing-dataset.txt" final 2 0.001
  withModelPath $ \path -> do
    saveModel path info trained
    (loaded, loadedTokenizer, loadedInfo) <- loadModel path
    assert "Metadata survives save/load" (loadedInfo == info)
    assert "Vocabulary IDs survive save/load"
      (tokenizerVocabulary loadedTokenizer == tokenizerVocabulary savedTokenizer &&
       all (\word -> tokenToId loadedTokenizer word == tokenToId savedTokenizer word)
         (tokenizerVocabulary savedTokenizer))
    let (before, _) = modelForward trained cfg x Nothing
        (after, _) = modelForward loaded (modelConfig loadedInfo) x Nothing
    assert "Reloaded weights produce identical logits"
      ((T.asValue (T.max (T.abs (before - after))) :: Float) == 0)
    response <- generate loaded cfg x 2
    assert "Reloaded model generates" (T.shape response == [2,6])
    overwrite <- try (saveModel path info model) :: IO (Either IOException ())
    assert "Existing model is protected" (either (const True) (const False) overwrite)
    -- Keep valid metadata but remove the model weights; loading must reject it.
    tensors <- Serialize.load path
    Serialize.save (take 1 tensors) path
    damaged <- try (loadModel path) :: IO (Either IOException (NanoChatModel, Tokenizer, ModelInfo))
    assert "Reject incomplete parameter archive" (either (const True) (const False) damaged)
  assert "Reject duplicate saved tokens"
    (either (const True) (const False) (tokenizerFromVocabulary
      (tokenizerVocabulary tokenizer ++ [Text.pack "hello"])))
  putStrLn "All NanoChat checks passed."

withModelPath :: (FilePath -> IO a) -> IO a
withModelPath = bracket acquire cleanup
  where
    acquire = do
      directory <- getTemporaryDirectory
      (path, handle) <- openBinaryTempFile directory "nanochat-test.model"
      hClose handle
      removeFile path
      pure path
    cleanup path = do
      exists <- doesFileExist path
      when exists (removeFile path)
