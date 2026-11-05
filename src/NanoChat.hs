{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module NanoChat where

import GHC.Generics (Generic)
import Torch
import qualified Torch.Functional as F
import qualified Torch.NN as NN

-- =====================================================================
-- 1. KONFIGURATION (Hyperparameter)
-- =====================================================================
data Config = Config
  { blockSize :: Int
  , nEmbd     :: Int
  , nHead     :: Int
  , nLayer    :: Int
  , vocabSize :: Int
  , dropout   :: Double
  } deriving (Show, Eq)

defaultConfig :: Config
defaultConfig = Config
  { blockSize = 128
  , nEmbd     = 128
  , nHead     = 4
  , nLayer    = 3
  , vocabSize = 100277 -- Entspricht cl100k_base (tiktoken)
  , dropout   = 0.1
  }

-- =====================================================================
-- 2. TRANSFORMATION-KOMPONENTEN (Attention & Blocks)
-- =====================================================================

-- Ein einzelner Attention-Kopf
data HeadSpec = HeadSpec { headSize :: Int, config :: Config }
data Head = Head
  { wQuery :: NN.Linear
  , wKey   :: NN.Linear
  , wValue :: NN.Linear
  } deriving (Generic)

instance NN.Parameterized Head

headForward :: Head -> Config -> Tensor -> Tensor
headForward Head{..} Config{..} x =
  let (b, t, c) = (size x 0, size x 1, size x 2)
      q = NN.linearForward wQuery x              -- (B, T, head_size)
      k = NN.linearForward wKey x                -- (B, T, head_size)
      v = NN.linearForward wValue x              -- (B, T, head_size)
      -- Skaliertes Skalarprodukt (Dot-Product Attention)
      scale = 1.0 / sqrt (fromIntegral (size k (-1)))
      scores = (q `matmul` transpose2d k) `mulScalar` scale -- (B, T, T)
      -- Kausale Maskierung (Tril)
      mask = F.tril (F.ones [t, t] (withDevice CPU))
      maskedScores = F.maskedFill scores (F.eq mask 0) (negate (1/0)) -- -inf
      weights = F.softmax maskedScores (-1)
  in weights `matmul` v                         -- (B, T, head_size)

-- Multi-Head Attention
data MultiHeadAttention = MultiHeadAttention
  { heads :: [Head]
  , proj  :: NN.Linear
  } deriving (Generic)

instance NN.Parameterized MultiHeadAttention

mhaForward :: MultiHeadAttention -> Config -> Tensor -> Tensor
mhaForward MultiHeadAttention{..} cfg x =
  let headOutputs = map (\h -> headForward h cfg x) heads
      concatHeads = F.cat (Dim (-1)) headOutputs
  in NN.linearForward proj concatHeads

-- Feed-Forward Netzwerk
data FeedForward = FeedForward
  { fc1 :: NN.Linear
  , fc2 :: NN.Linear
  } deriving (Generic)

instance NN.Parameterized FeedForward

ffwdForward :: FeedForward -> Tensor -> Tensor
ffwdForward FeedForward{..} x =
  let h = F.relu (NN.linearForward fc1 x)
  in NN.linearForward fc2 h

-- Ein kompletter Transformer Block
data Block = Block
  { sa   :: MultiHeadAttention
  , ffwd :: FeedForward
  , ln1  :: NN.LayerNorm
  , ln2  :: NN.LayerNorm
  } deriving (Generic)

instance NN.Parameterized Block

blockForward :: Block -> Config -> Tensor -> Tensor
blockForward Block{..} cfg x =
  let -- Residual-Verbindungen + LayerNorm separat vor den Operationen (Pre-LN)
      x1 = x + mhaForward sa cfg (NN.layerNormForward ln1 x)
      x2 = x1 + ffwdForward ffwd (NN.layerNormForward ln2 x1)
  in x2

-- =====================================================================
-- 3. DAS NANO-CHAT MODELL
-- =====================================================================
data NanoChatModel = NanoChatModel
  { tokenEmbedding    :: NN.Embedding
  , positionEmbedding :: NN.Embedding
  , blocks            :: [Block]
  , lnFinal           :: NN.LayerNorm
  , lmHead            :: NN.Linear
  } deriving (Generic)

instance NN.Parameterized NanoChatModel

-- Forward Pass für Verlustberechnung (Training)
modelForward :: NanoChatModel -> Config -> Tensor -> Maybe Tensor -> (Tensor, Maybe Tensor)
modelForward NanoChatModel{..} cfg@Config{..} idx mTargets =
  let (b, t) = (size idx 0, size idx 1)
      tokEmb = NN.embeddingForward tokenEmbedding idx
      posIdx = F.arange 0 t 1 (withDevice CPU)
      posEmb = NN.embeddingForward positionEmbedding posIdx
      
      -- Broadcast-Addition von Token- und Positions-Embeddings
      xStart = tokEmb + posEmb
      xEnd   = foldl (\x b -> blockForward b cfg x) xStart blocks
      xNorm  = NN.layerNormForward lnFinal xEnd
      logits = NN.linearForward lmHead xNorm -- (B, T, vocabSize)
      
      mLoss  = case mTargets of
        Nothing -> Nothing
        Just targets ->
          let c = size logits (-1)
              logits2d = F.view [b * t, c] logits
              targets1d = F.view [b * t] targets
          in Just (F.crossEntropyLoss logits2d targets1d)
  in (logits, mLoss)

-- Text-Generierungsschleife (Inferenz)
generate :: NanoChatModel -> Config -> Tensor -> Int -> IO Tensor
generate model cfg@Config{..} idx maxNewTokens = loop idx maxNewTokens
  where
    loop currentIdx 0 = return currentIdx
    loop currentIdx n = do
      -- Kontext auf maximale BlockSize kürzen
      let t = size currentIdx 1
          condIdx = if t > blockSize 
                    then F.slice 1 (t - blockSize) t 1 currentIdx 
                    else currentIdx
          (logits, _) = modelForward model cfg condIdx Nothing
          -- Letzten Zeitschritt extrahieren (B, C)
          lastLogits = F.select 1 (size logits 1 - 1) logits
          probs = F.softmax lastLogits (-1)
      -- Token-Sampling aus Wahrscheinlichkeitsverteilung
      nextIdx <- F.multinomial probs 1 True 
      let nextContext = F.cat (Dim 1) [currentIdx, nextIdx]
      loop nextContext (n - 1)
