module Main where

import Control.Monad (unless)
import qualified Data.Text as Text
import qualified Data.Text.IO as TextIO
import System.Environment (getArgs)
import System.Exit (die)
import System.IO (hFlush, isEOF, stdout)
import qualified Torch as T
import NanoChat.Transformer
import NanoChat.Checkpoint
import NanoChat.Tokenizer

main :: IO ()
main = do
  args <- getArgs
  path <- case args of
    [path] -> pure path
    _ -> die "Usage: nanochat MODEL_FILE"
  (model, tokenizer, info) <- loadModel path
  putStrLn ("Loaded " ++ path ++ " | steps " ++ show (completedSteps info)
    ++ " | last training loss " ++ show (finalTrainingLoss info))
  print (modelConfig info)
  putStrLn "Enter a message (empty line to quit)."
  chatLoop model (modelConfig info) tokenizer

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
