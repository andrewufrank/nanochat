module NanoChat.Training
  ( sampleBatch, trainStep, trainModel
  ) where

import Control.Monad (foldM, when)
import Torch (Tensor)
import qualified Torch as T
import qualified Torch.Optim as Optim
import NanoChat

sampleBatch :: Tensor -> Int -> Int -> IO (Tensor, Tensor)
-- | select batchSize (e.g. 8) randmom positios and takes at each position contextSize (e.g. 128) tokens (i.e. words)
sampleBatch allTokens batchSize contextSize
  | length (T.shape allTokens) /= 1 = ioError (userError "Training tokens must be a vector.")
  | batchSize <= 0 || contextSize <= 0 = ioError (userError "Batch and context sizes must be positive.")
  | total <= contextSize = ioError (userError "Dataset needs at least blockSize + 1 tokens.")
  | otherwise = do
      -- high is exclusive: include the last valid start, total-contextSize-1.
      starts <- T.randintIO 0 (total - contextSize) [batchSize] (T.withDType T.Int64 T.defaultOpts)
      let indices = T.asValue starts :: [Int]
          window offset i = T.sliceDim 0 (i + offset) (i + offset + contextSize) 1 allTokens
      pure (T.stack (T.Dim 0) (map (window 0) indices),
            T.stack (T.Dim 0) (map (window 1) indices))
  where total = T.size 0 allTokens

-- | Train with a fresh Adam optimizer, printing progress and returning the last batch loss.
-- Arguments are configuration, batch size, step count, dataset tokens, and initial model.
trainModel :: Config -> Int -> Int -> Tensor -> NanoChatModel
           -> IO (NanoChatModel, Float)
trainModel cfg batchSize steps allTokens model
  | steps <= 0 = ioError (userError "Training step count must be positive.")
  | otherwise = do
      let optimizer = Optim.mkAdam 0 0.9 0.999 (T.flattenParameters model)
      (trained, _, lastLoss) <-
        foldM (iteration cfg batchSize steps allTokens)
          (model, optimizer, 0) [1..steps]
      pure (trained, lastLoss)

-- Carry the updated model and optimizer state into the next iteration.
iteration :: Config -> Int -> Int -> Tensor
          -> (NanoChatModel, Optim.Adam, Float) -> Int
          -> IO (NanoChatModel, Optim.Adam, Float)
iteration cfg batchSize steps allTokens (current, opt, _) stepNumber = do
  (x, y) <- sampleBatch allTokens batchSize (blockSize cfg) 
  (updated, nextOpt, loss) <- trainStep current cfg opt x y
  when (stepNumber == 1 || stepNumber `mod` 50 == 0 || stepNumber == steps) $
    putStrLn ("Step " ++ show stepNumber ++ " | loss " ++ show loss)
  pure (updated, nextOpt, loss)

trainStep :: NanoChatModel -> Config -> Optim.Adam -> Tensor -> Tensor
          -> IO (NanoChatModel, Optim.Adam, Float)
trainStep model cfg optimizer inputs targets = do
  (_, lossMaybe) <- modelForwardTrain model cfg inputs (Just targets)
  case lossMaybe of
    Nothing -> ioError (userError "Training requires targets.")
    Just loss -> do
      let value = T.asValue loss :: Float
      if isNaN value || isInfinite value
        then ioError (userError "Training loss is not finite.")
        else do
          (updated, nextOptimizer) <- Optim.runStep model optimizer loss (T.asTensor (0.001 :: Float))
          pure (updated, nextOptimizer, value)
