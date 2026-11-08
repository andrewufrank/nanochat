-- | Sampling, optimization, and held-out evaluation.
--
-- Read "NanoChat.Tokenizer" first to see how words become IDs, then
-- 'sampleBatch' to see how those IDs become next-token examples. 'trainStep'
-- calls "NanoChat.Transformer" and Adam; 'trainModel' repeats that update.
-- 'evaluateLoss' measures predictions without updating any parameters.
--
-- Shapes use @B@ for batch size, @T@ for context length, and @N@ for the
-- number of dataset tokens. IDs are CPU Int64 tensors in the model vocabulary.
module NanoChat.Training
  ( sampleBatch, trainStep, trainModel, evaluateLoss
  ) where

import Control.Monad (foldM, when)
import Torch (Tensor)
import qualified Torch as T
import qualified Torch.Optim as Optim
import NanoChat.Transformer

-- | Sample one batch of @B@ windows, each containing @T@ input tokens.
-- Starting positions are chosen independently with replacement; windows may
-- overlap. This does not partition the dataset into epochs or groups of words.
--
-- Returns @(inputs, targets)@, both @[B,T]@. Targets are shifted one token
-- forward: @[a,b,c]@ predicts @[b,c,d]@. Requires @N >= T + 1@ and positive
-- batch/context sizes; invalid sizes raise an IO error.
sampleBatch :: Tensor -- ^ Dataset token IDs, shape @[N]@.
            -> Int    -- ^ Number of sequences in this batch, @B@.
            -> Int    -- ^ Tokens per input sequence, @T@.
            -> IO (Tensor, Tensor)
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

-- | Continue training with the supplied Adam state; return it for checkpointing.
-- Each step samples a new batch. Progress reports the loss of that batch with
-- training dropout enabled, before its parameter update; it is not validation
-- loss. The returned triple is @(updatedModel, updatedAdam, lastBatchLoss)@.
-- Adam's counter determines the cumulative step numbers when resuming.
--
-- 'foldM' passes each returned triple into the next 'iteration'. It does not
-- mutate the original Haskell bindings. Requires positive step count and rate.
-- Arguments: configuration, batch size, learning rate, additional step count,
-- dataset IDs @[N]@, model, and optimizer.
trainModel :: Config -> Int -> Float -> Int -> Tensor -> NanoChatModel -> Optim.Adam
           -> IO (NanoChatModel, Optim.Adam, Float)
trainModel cfg batchSize learningRate steps allTokens model optimizer
  | steps <= 0 = ioError (userError "Training step count must be positive.")
  | isNaN learningRate || isInfinite learningRate || learningRate <= 0 =
      ioError (userError "Learning rate must be finite and positive.")
  | Optim.iter optimizer < 0 || steps > maxBound - Optim.iter optimizer =
      ioError (userError "Invalid or overflowing training step count.")
  | otherwise =
      foldM (iteration cfg batchSize learningRate finalStep allTokens)
        (model, optimizer, 0) [firstStep..finalStep]
  where
    firstStep = Optim.iter optimizer + 1
    finalStep = Optim.iter optimizer + steps

-- Carry the updated model and optimizer state into the next iteration.
iteration :: Config -> Int -> Float -> Int -> Tensor
          -> (NanoChatModel, Optim.Adam, Float) -> Int
          -> IO (NanoChatModel, Optim.Adam, Float)
iteration cfg batchSize learningRate steps allTokens (current, opt, _) stepNumber = do
  (x, y) <- sampleBatch allTokens batchSize (blockSize cfg) 
  (updated, nextOpt, loss) <- trainStep current cfg learningRate opt x y
  when (stepNumber == 1 || stepNumber `mod` 50 == 0 || stepNumber == steps) $
    putStrLn ("Step " ++ show stepNumber ++ " | loss " ++ show loss)
  pure (updated, nextOpt, loss)

-- | One forward pass, gradient computation, and Adam update.
-- Inputs and targets have shape @[B,T]@, with targets shifted by one token.
-- The configuration and Adam state must match the model. Learning rate is
-- positive; IDs must be in vocabulary range and @1 <= T <= blockSize@.
-- Returns the new model, new optimizer state, and the mean cross-entropy
-- (natural-log units per token) measured /before/ this update. Dropout is on.
-- A non-finite loss raises an IO error instead of updating parameters.
trainStep :: NanoChatModel -> Config -> Float -> Optim.Adam -> Tensor -> Tensor
          -> IO (NanoChatModel, Optim.Adam, Float)
trainStep model cfg learningRate optimizer inputs targets = do
  (_, lossMaybe) <- modelForwardTrain model cfg inputs (Just targets)
  case lossMaybe of
    Nothing -> ioError (userError "Training requires targets.")
    Just loss -> do
      let value = T.asValue loss :: Float
      if isNaN value || isInfinite value
        then ioError (userError "Training loss is not finite.")
        else do
          (updated, nextOptimizer) <- Optim.runStep model optimizer loss (T.asTensor learningRate)
          pure (updated, nextOptimizer, value)

-- | Mean next-token cross-entropy on a fixed dataset, with dropout disabled.
-- No gradients are requested and no model or Adam update is performed.
--
-- Unlike 'sampleBatch', this visits the stream in order. Each token after the
-- first is a target exactly once. Context resets at windows of 'blockSize';
-- a final shorter window is included, and batch losses are weighted by their
-- target counts. Batch size changes grouping, not context boundaries.
-- Thus this is a fixed-window metric, not a full sliding-context likelihood.
--
-- Use a tokenizer built from /training/ text only. Unseen validation words map
-- to @[UNK]@; report their frequency when interpreting the score. Requires
-- at least two tokens, positive batch size, and a valid configuration.
evaluateLoss :: Config        -- ^ Configuration matching the model.
             -> Int           -- ^ Maximum windows per evaluation batch.
             -> Tensor        -- ^ Held-out token IDs, shape @[N]@, @N >= 2@.
             -> NanoChatModel -- ^ Model to evaluate without changing it.
             -> IO Float      -- ^ Mean negative log-probability per target token.
evaluateLoss cfg batchSize tokens model = do
  either (ioError . userError) pure (validateConfig cfg)
  when (batchSize <= 0) $ ioError (userError "Evaluation batch size must be positive.")
  when (length (T.shape tokens) /= 1) $ ioError (userError "Evaluation tokens must be a vector.")
  let targetCount = T.size 0 tokens - 1
  when (targetCount < 1) $ ioError (userError "Evaluation needs at least two tokens.")
  totalLoss <- go targetCount 0 0
  pure (realToFrac (totalLoss / fromIntegral targetCount))
  where
    go :: Int -> Int -> Double -> IO Double
    go targetCount start accumulated
      | start == targetCount = pure accumulated
      | otherwise = do
          let remaining = targetCount - start
              width = min (blockSize cfg) remaining
              count = min batchSize (remaining `div` width)
              starts = [start + i * width | i <- [0 .. count - 1]]
              window offset i = T.sliceDim 0 (i + offset) (i + offset + width) 1 tokens
              inputs = T.stack (T.Dim 0) (map (window 0) starts)
              targets = T.stack (T.Dim 0) (map (window 1) starts)
              (_, lossMaybe) = modelForward model cfg inputs (Just targets)
          loss <- maybe (ioError (userError "Evaluation requires targets.")) pure lossMaybe
          let value = T.asValue loss :: Float
              scored = count * width
              nextTotal = accumulated + realToFrac value * fromIntegral scored
          when (isNaN value || isInfinite value) $
            ioError (userError "Evaluation loss is not finite.")
          -- Force the scalar accumulator so previous tensor graphs can be released.
          nextTotal `seq` go targetCount (start + scored) nextTotal
