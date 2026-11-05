module Main where

import Control.Monad (foldM, unless, when)
import qualified Data.Text as Text
import qualified Data.Text.IO as TextIO
import System.Environment (getArgs)
import System.Exit (die)
import System.IO (hFlush, isEOF, stdout)
import Text.Read (readMaybe)
import qualified Torch as T
import qualified Torch.Optim as Optim
import NanoChat
import NanoChat.Training

main :: IO ()
main = do
  args <- getArgs
  (path, steps) <- case args of
    [] -> pure ("chat_dataset.txt", 500)
    [path] -> pure (path, 500)
    [path, count] | Just n <- readMaybe count, n > 0 -> pure (path, n)
    _ -> die "Usage: nanochat [dataset.txt [positive-training-steps]]"
  raw <- TextIO.readFile path
  let tokenizer = buildTokenizer raw
      encoded = map (tokenToId tokenizer) (Text.words raw)
      cfg = defaultConfig { vocabSize = tokenizerSize tokenizer }
      allTokens = T.asTensor encoded
  when (length encoded <= blockSize cfg) $
    die ("Dataset needs at least " ++ show (blockSize cfg + 1) ++ " whitespace-separated tokens.")
  putStrLn ("CPU training: " ++ show (length encoded) ++ " tokens, "
    ++ show (vocabSize cfg) ++ " vocabulary entries, " ++ show steps ++ " steps.")
  model <- T.sample (NanoChatModelSpec cfg)
  let optimizer = Optim.mkAdam 0 0.9 0.999 (T.flattenParameters model)
      iteration (current, opt) stepNumber = do
        (x, y) <- getBatch allTokens 8 (blockSize cfg)
        (updated, nextOpt, loss) <- trainStep current cfg opt x y
        when (stepNumber == 1 || stepNumber `mod` 50 == 0 || stepNumber == steps) $
          putStrLn ("Step " ++ show stepNumber ++ " | loss " ++ show loss)
        pure (updated, nextOpt)
  (trained, _) <- foldM iteration (model, optimizer) [1..steps :: Int]
  putStrLn "Training finished. Enter a message (empty line to quit)."
  chatLoop trained cfg tokenizer

chatLoop :: NanoChatModel -> Config -> Tokenizer -> IO ()
chatLoop model cfg tokenizer = do
  putStr "\n[USER]: "
  hFlush stdout
  eof <- isEOF
  unless eof $ do
    input <- TextIO.getLine
    unless (Text.null (Text.strip input)) $ do
      let prompt = Text.words ("[USER] " <> input <> " [BOT]")
          ids = map (tokenToId tokenizer) prompt
          context = T.asTensor [ids]
      result <- generate model cfg context 30
      let generated = drop (length ids) (T.asValue (T.select 0 0 result) :: [Int])
          wordsOut = takeWhile (/= "[END]") (map (idToToken tokenizer) generated)
      TextIO.putStrLn ("[BOT]: " <> Text.unwords wordsOut)
      chatLoop model cfg tokenizer
