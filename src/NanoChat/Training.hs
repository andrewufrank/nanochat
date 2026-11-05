module NanoChat.Training
  ( Tokenizer, buildTokenizer, tokenToId, idToToken, tokenizerSize
  , getBatch, trainStep
  ) where

import qualified Data.Map.Strict as Map
import qualified Data.IntMap.Strict as IntMap
import qualified Data.Set as Set
import qualified Data.Text as Text
import Torch (Tensor)
import qualified Torch as T
import qualified Torch.Optim as Optim
import NanoChat

-- A word tokenizer, not BPE. Reserve a real unknown token and chat markers.
data Tokenizer = Tokenizer (Map.Map Text.Text Int) (IntMap.IntMap Text.Text)

buildTokenizer :: Text.Text -> Tokenizer
buildTokenizer input = Tokenizer (Map.fromList entries)
  (IntMap.fromList [(i, word) | (word, i) <- entries])
  where
    reserved = map Text.pack ["[UNK]", "[USER]", "[BOT]", "[END]"]
    wordsOnly = Set.fromList (Text.words input) `Set.difference` Set.fromList reserved
    entries = zip (reserved ++ Set.toAscList wordsOnly) [0..]

tokenToId :: Tokenizer -> Text.Text -> Int
tokenToId (Tokenizer forward _) word = Map.findWithDefault 0 word forward

idToToken :: Tokenizer -> Int -> Text.Text
idToToken (Tokenizer _ backward) i = IntMap.findWithDefault (Text.pack "[UNK]") i backward

tokenizerSize :: Tokenizer -> Int
tokenizerSize (Tokenizer forward _) = Map.size forward

getBatch :: Tensor -> Int -> Int -> IO (Tensor, Tensor)
getBatch allTokens batchSize contextSize
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
