module Main where

import Control.Exception (IOException, try)
import Control.Monad (foldM, unless)
import qualified Data.Text as Text
import qualified Torch as T
import qualified Torch.Optim as Optim
import NanoChat
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
  (bx, by) <- getBatch sequenceTokens 2 4
  assert "Minimal-length dataset and shifted targets"
    (T.asValue bx == ([[0,1,2,3],[0,1,2,3]] :: [[Int]]) &&
     T.asValue by == ([[1,2,3,4],[1,2,3,4]] :: [[Int]]))
  badBatch <- try (getBatch sequenceTokens 2 5) :: IO (Either IOException (T.Tensor, T.Tensor))
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
  putStrLn "All NanoChat checks passed."
