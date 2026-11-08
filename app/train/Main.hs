module Main where

import Control.Monad (when)
import Data.List (isPrefixOf)
import qualified Data.Text as Text
import qualified Data.Text.IO as TextIO
import System.Environment (getArgs)
import System.Exit (die)
import Text.Read (readMaybe)
import qualified Torch as T
import qualified Torch.Optim as Optim
import NanoChat.Transformer
import NanoChat.Checkpoint
import NanoChat.Tokenizer
import NanoChat.Training

batchSize :: Int
batchSize = 8

learningRate :: Float
learningRate = 0.001

-- A resume run writes a new checkpoint, preserving its input checkpoint.
data Command = Fresh FilePath FilePath Int | Resume FilePath FilePath FilePath Int

-- Optional flags precede the existing fresh/resume command.
parseOptions :: [String] -> Either String (Bool, Maybe FilePath, Command)
parseOptions = go False Nothing
  where
    go False validation ("--tiny" : rest) = go True validation rest
    go tiny Nothing ("--validation" : path : rest)
      | not ("--" `isPrefixOf` path) = go tiny (Just path) rest
    go tiny validation args = do
      command <- parseCommand args
      case command of
        Resume _ _ _ _ | tiny -> Left "--tiny applies only to new runs; resume uses the saved configuration."
        _ -> Right (tiny, validation, command)

parseCommand :: [String] -> Either String Command
parseCommand args = case args of
  ["--resume", input, output, path] -> Right (Resume input output path 500)
  ["--resume", input, output, path, count]
    | Just n <- readMaybe count, n > 0 -> Right (Resume input output path n)
  [output, path] | not ("--" `isPrefixOf` output) -> Right (Fresh output path 500)
  [output, path, count]
    | not ("--" `isPrefixOf` output), Just n <- readMaybe count, n > 0 -> Right (Fresh output path n)
  _ -> Left $ unlines
    [ "Usage: nanochat-train [--tiny] [--validation FILE] MODEL_FILE DATASET [positive-training-steps]"
    , "       nanochat-train [--validation FILE] --resume INPUT_MODEL OUTPUT_MODEL DATASET [positive-additional-steps]"
    ]

main :: IO ()
main = do
  (tiny, validationPath, command) <- getArgs >>= either die pure . parseOptions
  let (output, path, steps) = case command of
        Fresh destination dataset count -> (destination, dataset, count)
        Resume _ destination dataset count -> (destination, dataset, count)
  ensureNewModelFile output
  raw <- TextIO.readFile path
  (model, tokenizer, info, optimizer) <- case command of
    Fresh _ _ _ -> do
      let tokenizer = buildTokenizer raw
          preset = if tiny then tinyConfig else defaultConfig
          cfg = preset { vocabSize = tokenizerSize tokenizer }
      model <- T.sample (NanoChatModelSpec cfg)
      let optimizer = Optim.mkAdam 0 0.9 0.999 (T.flattenParameters model)
          info = ModelInfo 2 cfg (tokenizerVocabulary tokenizer) 0 path 0 batchSize learningRate
      pure (model, tokenizer, info, optimizer)
    Resume input _ _ _ -> loadTrainingCheckpoint input

  -- Resuming must retain the original token IDs, even on a different dataset.
  let wordsInDataset = Text.words raw
      encoded = map (tokenToId tokenizer) wordsInDataset
      cfg = modelConfig info
      allTokens = T.asTensor encoded
      unknownCount = length (filter (== tokenToId tokenizer (Text.pack "[UNK]")) encoded)
  when (length encoded <= blockSize cfg) $
    die ("Dataset needs at least " ++ show (blockSize cfg + 1) ++ " whitespace-separated tokens.")
  putStrLn ("CPU training: " ++ show (length encoded) ++ " tokens, "
    ++ show (vocabSize cfg) ++ " vocabulary entries, " ++ show steps ++ " additional steps; "
    ++ show (completedSteps info) ++ " already completed.")
  when (unknownCount > 0) $
    putStrLn (show unknownCount ++ " dataset tokens encoded as [UNK] using the saved vocabulary.")
  print cfg

  -- Validation text is encoded with the training vocabulary, never fitted to it.
  validation <- case validationPath of
    Nothing -> pure Nothing
    Just validationFile -> do
      text <- TextIO.readFile validationFile
      let ids = map (tokenToId tokenizer) (Text.words text)
          unknown = length (filter (== tokenToId tokenizer (Text.pack "[UNK]")) ids)
      when (length ids < 2) $ die "Validation dataset needs at least two tokens."
      putStrLn ("Validation: " ++ show (length ids) ++ " tokens; " ++ show unknown ++ " encoded as [UNK].")
      pure (Just (T.asTensor ids))
  reportEvaluation "Before training" cfg (trainingBatchSize info) allTokens validation model

  (trained, nextOptimizer, lastLoss) <- trainModel cfg (trainingBatchSize info)
    (trainingLearningRate info) steps allTokens model optimizer
  reportEvaluation "After training" cfg (trainingBatchSize info) allTokens validation trained
  let nextInfo = info
        { formatVersion = 2, completedSteps = Optim.iter nextOptimizer
        , datasetPath = path, finalTrainingLoss = lastLoss }
  saveTrainingCheckpoint output nextInfo trained nextOptimizer
  putStrLn ("Saved training checkpoint to " ++ output)
  putStrLn ("Chat with: cabal run nanochat -- " ++ show output)

-- Compare both datasets in evaluation mode, so dropout is not a confounding factor.
-- Without --validation, retain the inexpensive training-only behavior.
reportEvaluation :: String -> Config -> Int -> T.Tensor -> Maybe T.Tensor -> NanoChatModel -> IO ()
reportEvaluation _ _ _ _ Nothing _ = pure ()
reportEvaluation label cfg size trainingTokens (Just validationTokens) model = do
  trainingLoss <- evaluateLoss cfg size trainingTokens model
  validationLoss <- evaluateLoss cfg size validationTokens model
  putStrLn (label ++ " | training-set evaluation loss " ++ show trainingLoss
    ++ " | validation loss " ++ show validationLoss ++ " (nats/token; dropout off)")
