module Main where

import Control.Exception (IOException, bracket, try)
import Control.Monad (foldM, forM_, unless, when)
import qualified Data.Text as Text
import qualified Torch as T
import qualified Torch.Optim as Optim
import qualified Torch.Serialize as Serialize
import System.Directory (doesFileExist, getTemporaryDirectory, removeFile)
import System.IO (hClose, openBinaryTempFile)
import NanoChat.Transformer
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
      (updated, next, _) <- trainStep current cfg 0.001 opt x y
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
  -- Evaluation covers each target once, including an uneven final window.
  -- Batch size must affect grouping only, and configured dropout must stay off.
  let heldOut = T.asTensor ([0,1,2,3,4,5,0,1,2,3] :: [Int])
      windowLoss inputs targets = do
        let (_, result) = modelForward trained evalCfg
              (T.asTensor [inputs :: [Int]]) (Just (T.asTensor [targets :: [Int]]))
        tensor <- maybe (ioError (userError "Missing evaluation loss")) pure result
        pure (T.asValue tensor :: Float)
  valA <- evaluateLoss evalCfg 2 heldOut trained
  valB <- evaluateLoss evalCfg 2 heldOut trained
  valSingle <- evaluateLoss evalCfg 1 heldOut trained
  firstWindow <- windowLoss [0,1,2,3] [1,2,3,4]
  secondWindow <- windowLoss [4,5,0,1] [5,0,1,2]
  finalWindow <- windowLoss [2] [3]
  let expectedValidation = (4 * firstWindow + 4 * secondWindow + finalWindow) / 9
  assert "Validation weights all targets, including short final window"
    (abs (valA - expectedValidation) < 1e-5)
  assert "Validation is deterministic with dropout configured" (valA == valB)
  assert "Validation is independent of batch grouping" (abs (valA - valSingle) < 1e-5)
  shortValidation <- evaluateLoss evalCfg 8 (T.asTensor ([2,3] :: [Int])) trained
  assert "Validation accepts a single prediction" (abs (shortValidation - finalWindow) < 1e-5)
  forM_ [T.asTensor ([] :: [Int]), T.asTensor ([0] :: [Int]), x] $ \badTokens -> do
    invalid <- try (evaluateLoss evalCfg 2 badTokens trained) :: IO (Either IOException Float)
    assert "Validation rejects short or non-vector input" (either (const True) (const False) invalid)
  badEvalBatch <- try (evaluateLoss evalCfg 0 heldOut trained) :: IO (Either IOException Float)
  assert "Validation rejects zero batch size" (either (const True) (const False) badEvalBatch)
  badEvalConfig <- try (evaluateLoss (evalCfg {blockSize = 0}) 2 heldOut trained) :: IO (Either IOException Float)
  assert "Validation rejects invalid configuration" (either (const True) (const False) badEvalConfig)
  assert "Tiny example configuration is valid" (validateConfig tinyConfig == Right ())
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
    legacyResume <- try (loadTrainingCheckpoint path)
      :: IO (Either IOException (NanoChatModel, Tokenizer, ModelInfo, Optim.Adam))
    assert "Legacy checkpoints cannot silently restart Adam"
      (either (const True) (const False) legacyResume)
    -- Keep valid metadata but remove the model weights; loading must reject it.
    tensors <- Serialize.load path
    Serialize.save (take 1 tensors) path
    damaged <- try (loadModel path) :: IO (Either IOException (NanoChatModel, Tokenizer, ModelInfo))
    assert "Reject incomplete parameter archive" (either (const True) (const False) damaged)
  assert "Reject duplicate saved tokens"
    (either (const True) (const False) (tokenizerFromVocabulary
      (tokenizerVocabulary tokenizer ++ [Text.pack "hello"])))
  -- A single possible batch and disabled dropout make the comparison deterministic,
  -- without depending on global RNG-state capture. Use non-default Adam settings/rate.
  let resumeRate = 0.0003
      initialAdam = Optim.mkAdam 0 0.8 0.95 (T.flattenParameters model)
  (wholeModel, wholeAdam, wholeLoss) <- trainModel cfg 2 resumeRate 5 sequenceTokens model initialAdam
  (partModel, partAdam, partLoss) <- trainModel cfg 2 resumeRate 2 sequenceTokens model initialAdam
  let resumeInfo = ModelInfo 2 cfg (tokenizerVocabulary tokenizer) 2 "original.txt" partLoss 2 resumeRate
  withModelPath $ \path -> do
    saveTrainingCheckpoint path resumeInfo partModel partAdam
    (loaded, loadedTokenizer, loadedInfo, loadedAdam) <- loadTrainingCheckpoint path
    assert "Training metadata and vocabulary survive checkpoint"
      (loadedInfo == resumeInfo && tokenizerVocabulary loadedTokenizer == tokenizerVocabulary tokenizer)
    assert "Adam scalar state survives checkpoint"
      (Optim.iter loadedAdam == 2 && Optim.beta1 loadedAdam == 0.8 && Optim.beta2 loadedAdam == 0.95)
    assert "Adam moment tensors survive checkpoint"
      (sameTensors (Optim.m1 partAdam ++ Optim.m2 partAdam) (Optim.m1 loadedAdam ++ Optim.m2 loadedAdam))
    (chatModel, _, _) <- loadModel path
    assert "Chat loads version-2 weights" (sameTensors (weights partModel) (weights chatModel))
    (resumed, resumedAdam, resumedLoss) <- trainModel (modelConfig loadedInfo)
      (trainingBatchSize loadedInfo) (trainingLearningRate loadedInfo) 3 sequenceTokens loaded loadedAdam
    assert "Resumed training matches uninterrupted weights and loss"
      (sameTensors (weights wholeModel) (weights resumed) && wholeLoss == resumedLoss)
    assert "Resumed training matches uninterrupted optimizer"
      (Optim.iter resumedAdam == 5 &&
       sameTensors (Optim.m1 wholeAdam ++ Optim.m2 wholeAdam) (Optim.m1 resumedAdam ++ Optim.m2 resumedAdam))
    overwrite <- try (saveTrainingCheckpoint path resumeInfo partModel partAdam) :: IO (Either IOException ())
    assert "Training checkpoint overwrite is refused" (either (const True) (const False) overwrite)
    tensors <- Serialize.load path
    Serialize.save (init tensors) path
    damaged <- try (loadTrainingCheckpoint path)
      :: IO (Either IOException (NanoChatModel, Tokenizer, ModelInfo, Optim.Adam))
    assert "Reject missing Adam moment tensor" (either (const True) (const False) damaged)
    Serialize.save (init tensors ++ [T.zeros' [1]]) path
    wrongShape <- try (loadTrainingCheckpoint path)
      :: IO (Either IOException (NanoChatModel, Tokenizer, ModelInfo, Optim.Adam))
    assert "Reject wrong Adam moment shape" (either (const True) (const False) wrongShape)
  withModelPath $ \path -> do
    badCounter <- try (saveTrainingCheckpoint path resumeInfo partModel (partAdam {Optim.iter = 1}))
      :: IO (Either IOException ())
    assert "Reject inconsistent optimizer step count" (either (const True) (const False) badCounter)
    badBeta <- try (saveTrainingCheckpoint path resumeInfo partModel (partAdam {Optim.beta1 = 1}))
      :: IO (Either IOException ())
    assert "Reject invalid Adam beta" (either (const True) (const False) badBeta)
    exists <- doesFileExist path
    assert "Rejected checkpoint does not create output" (not exists)
  putStrLn "All NanoChat checks passed."

weights :: NanoChatModel -> [T.Tensor]
weights = map T.toDependent . T.flattenParameters

sameTensors :: [T.Tensor] -> [T.Tensor] -> Bool
sameTensors left right = length left == length right && and (zipWith same left right)
  where
    same a b = T.shape a == T.shape b && T.dtype a == T.dtype b &&
      (T.asValue (T.max (T.abs (a - b))) :: Float) == 0


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
