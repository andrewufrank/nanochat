-- | A whitespace tokenizer with a fixed word-to-ID vocabulary.
--
-- Build it from training text, then reuse it for validation, generation, and
-- resumed training. Punctuation and case are preserved: @hello@ and @hello.@
-- are different tokens. This is deliberately simpler than a subword tokenizer.
-- Next read @sampleBatch@ in "NanoChat.Training" to turn IDs into examples.
module NanoChat.Tokenizer
  ( Tokenizer, buildTokenizer, tokenToId, idToToken, tokenizerSize
  , tokenizerVocabulary, tokenizerFromVocabulary
  ) where

import qualified Data.Map.Strict as Map
import qualified Data.IntMap.Strict as IntMap
import qualified Data.Set as Set
import qualified Data.Text as Text

-- | Forward and reverse vocabulary maps. The constructor is hidden so callers
-- build a consistent mapping with 'buildTokenizer' or 'tokenizerFromVocabulary'.
data Tokenizer = Tokenizer (Map.Map Text.Text Int) (IntMap.IntMap Text.Text)

-- | Reserve IDs 0–3 for @[UNK]@, @[USER]@, @[BOT]@, and @[END]@, then assign
-- ascending IDs to the distinct remaining words in sorted text order.
-- Repetition does not enlarge the vocabulary. Use training text only.
--
-- >>> let tok = buildTokenizer (Text.pack "hello hello world")
-- >>> tokenizerSize tok
-- 6
-- >>> tokenToId tok (Text.pack "hello")
-- 4
buildTokenizer :: Text.Text -> Tokenizer
buildTokenizer input = Tokenizer (Map.fromList entries)
  (IntMap.fromList [(i, word) | (word, i) <- entries])
  where
    reserved = map Text.pack ["[UNK]", "[USER]", "[BOT]", "[END]"]
    wordsOnly = Set.fromList (Text.words input) `Set.difference` Set.fromList reserved
    entries = zip (reserved ++ Set.toAscList wordsOnly) [0..]

-- | Look up a word; words absent from the vocabulary map to ID 0 (@[UNK]@).
-- The input is one token, not a sentence to be split.
tokenToId :: Tokenizer -> Text.Text -> Int
tokenToId (Tokenizer forward _) word = Map.findWithDefault 0 word forward

-- | Look up an ID; unknown or negative IDs return the text @[UNK]@.
idToToken :: Tokenizer -> Int -> Text.Text
idToToken (Tokenizer _ backward) i = IntMap.findWithDefault (Text.pack "[UNK]") i backward

-- | Count all vocabulary entries, including the four reserved tokens.
tokenizerSize :: Tokenizer -> Int
tokenizerSize (Tokenizer forward _) = Map.size forward

-- | List words in token-ID order for checkpoints. Preserving this order is
-- essential: changing IDs would change the meaning of learned embedding rows.
tokenizerVocabulary :: Tokenizer -> [Text.Text]
tokenizerVocabulary (Tokenizer _ backward) = IntMap.elems backward

-- | Restore exactly the supplied token-ID order. Reject missing/reordered
-- reserved tokens, duplicate words, or empty words. This is the inverse of
-- 'tokenizerVocabulary' for a valid tokenizer; it does not sort the input.
tokenizerFromVocabulary :: [Text.Text] -> Either String Tokenizer
tokenizerFromVocabulary wordsInOrder
  | take 4 wordsInOrder /= map Text.pack ["[UNK]", "[USER]", "[BOT]", "[END]"] =
      Left "Saved vocabulary has invalid reserved tokens."
  | Set.size (Set.fromList wordsInOrder) /= length wordsInOrder =
      Left "Saved vocabulary contains duplicate tokens."
  | any Text.null wordsInOrder = Left "Saved vocabulary contains an empty token."
  | otherwise = Right (Tokenizer (Map.fromList (zip wordsInOrder [0..]))
      (IntMap.fromList (zip [0..] wordsInOrder)))
