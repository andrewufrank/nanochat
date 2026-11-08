module NanoChat.Tokenizer
  ( Tokenizer, buildTokenizer, tokenToId, idToToken, tokenizerSize
  , tokenizerVocabulary, tokenizerFromVocabulary
  ) where

import qualified Data.Map.Strict as Map
import qualified Data.IntMap.Strict as IntMap
import qualified Data.Set as Set
import qualified Data.Text as Text

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

-- Store vocabulary in token-ID order, never rebuild it from the chat input.
tokenizerVocabulary :: Tokenizer -> [Text.Text]
tokenizerVocabulary (Tokenizer _ backward) = IntMap.elems backward

tokenizerFromVocabulary :: [Text.Text] -> Either String Tokenizer
tokenizerFromVocabulary wordsInOrder
  | take 4 wordsInOrder /= map Text.pack ["[UNK]", "[USER]", "[BOT]", "[END]"] =
      Left "Saved vocabulary has invalid reserved tokens."
  | Set.size (Set.fromList wordsInOrder) /= length wordsInOrder =
      Left "Saved vocabulary contains duplicate tokens."
  | any Text.null wordsInOrder = Left "Saved vocabulary contains an empty token."
  | otherwise = Right (Tokenizer (Map.fromList (zip wordsInOrder [0..]))
      (IntMap.fromList (zip [0..] wordsInOrder)))

