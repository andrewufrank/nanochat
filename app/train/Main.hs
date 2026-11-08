module Main where

import Control.Monad (when)
import qualified Data.Text as Text
import qualified Data.Text.IO as TextIO
import System.Directory (createDirectoryIfMissing)
import System.Environment (getArgs)
import System.Exit (die)
import System.FilePath (takeDirectory)
import Text.Read (readMaybe)
import qualified Torch as T
import NanoChat
import NanoChat.Checkpoint
import NanoChat.Tokenizer
import NanoChat.Training

batchSize :: Int
batchSize = 8

main :: IO ()
main = do
    -- read name of dataset and number of steps
    args <- getArgs
    (output, path, steps) <- case args of
        [output, path] -> pure (output, path, 500)
        [output, path, count] | Just n <- readMaybe count, n > 0 -> pure (output, path, n)
        _ -> die "Usage: nanochat-train MODEL_FILE DATASET [positive-training-steps]"
    ensureNewModelFile output
    createDirectoryIfMissing True (takeDirectory output)

        -- read dataset, tokenize and create encoder (see Tokenizer.hs)
    raw <- TextIO.readFile path
    let tokenizer = buildTokenizer raw
        encoded = map (tokenToId tokenizer) (Text.words raw)
        cfg = defaultConfig { vocabSize = tokenizerSize tokenizer }
        allTokens = T.asTensor encoded

        -- check configuration
    when (length encoded <= blockSize cfg) $
        die ("Dataset needs at least " ++ show (blockSize cfg + 1) ++ " whitespace-separated tokens.")
    putStrLn ("CPU training: " ++ show (length encoded) ++ " tokens, "
        ++ show (vocabSize cfg) ++ " vocabulary entries, " ++ show steps ++ " steps.")
    print cfg

        -- initialize model with random numbers, then train it
    model <- T.sample (NanoChatModelSpec cfg)
    (trained, lastLoss) <- trainModel cfg batchSize steps allTokens model
    let info = ModelInfo
            { formatVersion = 1, modelConfig = cfg, vocabulary = tokenizerVocabulary tokenizer
            , completedSteps = steps, datasetPath = path, finalTrainingLoss = lastLoss
            , trainingBatchSize = batchSize, trainingLearningRate = 0.001 }

        -- training completed, save model and show how to start the chat
    saveModel output info trained
    putStrLn ("Saved model to " ++ output)
    putStrLn ("Chat with: cabal run nanochat -- " ++ show output)
