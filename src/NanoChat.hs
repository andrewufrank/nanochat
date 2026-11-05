{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

-- | A small CPU transformer inspired by NanoChat; not checkpoint compatible.
module NanoChat where

import Control.Monad (foldM, replicateM)
import Data.Functor.Identity (Identity (..))
import GHC.Generics (Generic)
import Torch (Tensor, Parameter, Dim (..), KeepDim (..))
import qualified Torch as T
import qualified Torch.Functional as F
import qualified Torch.NN as NN

data Config = Config
  { blockSize :: Int, nEmbd :: Int, nHead :: Int, nLayer :: Int
  , vocabSize :: Int, dropout :: Double
  } deriving (Show, Eq)

defaultConfig :: Config
defaultConfig = Config 128 128 4 3 256 0.1

validateConfig :: Config -> Either String ()
validateConfig Config{..}
  | any (<= 0) [blockSize, nEmbd, nHead, nLayer, vocabSize] =
      Left "Model dimensions must be positive."
  | nEmbd `mod` nHead /= 0 = Left "nEmbd must be divisible by nHead."
  | isNaN dropout || dropout < 0 || dropout >= 1 = Left "dropout must be in [0, 1)."
  | otherwise = Right ()

data HeadSpec = HeadSpec { headSize :: Int, config :: Config }
data Head = Head
  { wQuery :: NN.Linear, wKey :: NN.Linear, wValue :: NN.Linear }
  deriving (Generic)
instance NN.Parameterized Head
instance NN.Randomizable HeadSpec Head where
  sample (HeadSpec width cfg) = Head <$> linear <*> linear <*> linear
    where linear = NN.sample (NN.LinearSpec (nEmbd cfg) width)

data MultiHeadAttention = MultiHeadAttention
  { heads :: [Head], proj :: NN.Linear } deriving (Generic)
instance NN.Parameterized MultiHeadAttention

data FeedForward = FeedForward
  { fc1 :: NN.Linear, fc2 :: NN.Linear } deriving (Generic)
instance NN.Parameterized FeedForward

-- Explicitly register the untyped layers' parameters for autograd and Adam.
newtype Embedding = Embedding Parameter deriving (Generic)
instance NN.Parameterized Embedding

embeddingForward :: Embedding -> Tensor -> Tensor
embeddingForward (Embedding weights) = F.embedding' (T.toDependent weights)

data LayerNorm = LayerNorm Parameter Parameter deriving (Generic)
instance NN.Parameterized LayerNorm

sampleLayerNorm :: Int -> IO LayerNorm
sampleLayerNorm width = LayerNorm
  <$> T.makeIndependent (T.ones' [width])
  <*> T.makeIndependent (T.zeros' [width])

layerNormForward :: LayerNorm -> Tensor -> Tensor
layerNormForward (LayerNorm gain bias) x =
  let meanLast = F.meanDim (Dim (-1)) KeepDim (T.dtype x)
      centered = x - meanLast x
      variance = meanLast (centered * centered)
  in centered / F.sqrt (variance + 1e-5) * T.toDependent gain + T.toDependent bias

data Block = Block
  { sa :: MultiHeadAttention, ffwd :: FeedForward
  , ln1 :: LayerNorm, ln2 :: LayerNorm } deriving (Generic)
instance NN.Parameterized Block

data NanoChatModel = NanoChatModel
  { tokenEmbedding :: Embedding, positionEmbedding :: Embedding
  , blocks :: [Block], lnFinal :: LayerNorm, lmHead :: NN.Linear
  } deriving (Generic)
instance NN.Parameterized NanoChatModel

newtype NanoChatModelSpec = NanoChatModelSpec Config
instance NN.Randomizable NanoChatModelSpec NanoChatModel where
  sample (NanoChatModelSpec cfg@Config{..}) = do
    either (ioError . userError) pure (validateConfig cfg)
    let embedding count = do
          weights <- T.randnIO' [count, nEmbd]
          Embedding <$> T.makeIndependent (F.mulScalar (0.02 :: Float) weights)
        linear a b = NN.sample (NN.LinearSpec a b)
        block = Block
          <$> (MultiHeadAttention
            <$> replicateM nHead (NN.sample (HeadSpec (nEmbd `div` nHead) cfg))
            <*> linear nEmbd nEmbd)
          <*> (FeedForward <$> linear nEmbd (4 * nEmbd) <*> linear (4 * nEmbd) nEmbd)
          <*> sampleLayerNorm nEmbd <*> sampleLayerNorm nEmbd
    NanoChatModel <$> embedding vocabSize <*> embedding blockSize
      <*> replicateM nLayer block <*> sampleLayerNorm nEmbd <*> linear nEmbd vocabSize

-- | Deterministic evaluation (dropout disabled).
modelForward :: NanoChatModel -> Config -> Tensor -> Maybe Tensor -> (Tensor, Maybe Tensor)
modelForward model cfg idx targets =
  runIdentity (modelForwardWith Identity model cfg idx targets)

-- | Training uses stochastic dropout; evaluation and generation do not.
modelForwardTrain :: NanoChatModel -> Config -> Tensor -> Maybe Tensor -> IO (Tensor, Maybe Tensor)
modelForwardTrain model cfg = modelForwardWith (F.dropout (dropout cfg) True) model cfg

modelForwardWith :: Monad m => (Tensor -> m Tensor) -> NanoChatModel -> Config
                 -> Tensor -> Maybe Tensor -> m (Tensor, Maybe Tensor)
modelForwardWith dropTensor NanoChatModel{..} Config{..} idx targets
  | length (T.shape idx) /= 2 = error "Input must have shape [batch, time]."
  | t < 1 || t > blockSize = error "Input length must be between 1 and blockSize."
  | otherwise = do
      let positions = T.asTensor ([0 .. t - 1] :: [Int])
      x <- dropTensor (embeddingForward tokenEmbedding idx + embeddingForward positionEmbedding positions)
      hidden <- foldM blockForward x blocks
      let logits = NN.linearForward lmHead (layerNormForward lnFinal hidden)
          loss y
            | T.shape y /= T.shape idx = error "Targets must have the same shape as inputs."
            | otherwise = F.nllLoss' (T.reshape [b * t] y)
                (F.logSoftmax (Dim 1) (T.reshape [b * t, vocabSize] logits))
      pure (logits, fmap loss targets)
  where
    b = T.size 0 idx
    t = T.size 1 idx
    headForward Head{..} x = do
      let q = NN.linearForward wQuery x
          k = NN.linearForward wKey x
          v = NN.linearForward wValue x
          scale = 1 / sqrt (fromIntegral (T.size (-1) k) :: Float)
          scores = F.mulScalar scale (F.matmul q (F.transpose (Dim (-2)) (Dim (-1)) k))
          -- Additive [time,time] mask broadcasts across the batch.
          mask = T.asTensor
            [[if j > i then negate (1 / 0) else 0 :: Float | j <- [0 .. t - 1]]
            | i <- [0 .. t - 1]]
          masked = scores + mask
      weights <- dropTensor (F.softmax (Dim (-1)) masked)
      pure (F.matmul weights v)
    blockForward x Block{..} = do
      hs <- mapM (\h -> headForward h (layerNormForward ln1 x)) (heads sa)
      attention <- dropTensor (NN.linearForward (proj sa) (F.cat (Dim (-1)) hs))
      let residual = x + attention
          FeedForward{..} = ffwd
          feed = NN.linearForward fc2 (F.relu (NN.linearForward fc1 (layerNormForward ln2 residual)))
      output <- dropTensor feed
      pure (residual + output)

-- | Autoregressive sampling, cropping the context to the model's window.
generate :: NanoChatModel -> Config -> Tensor -> Int -> IO Tensor
generate model cfg idx count
  | count < 0 = ioError (userError "Number of new tokens must be nonnegative.")
  | otherwise = loop idx count
  where
    loop current 0 = pure current
    loop current n = do
      let t = T.size 1 current
          context = T.sliceDim 1 (max 0 (t - blockSize cfg)) t 1 current
          (logits, _) = modelForward model cfg context Nothing
      -- Sampling does not need to retain an autograd graph.
      lastLogits <- F.detach (T.select 1 (T.size 1 logits - 1) logits)
      next <- T.multinomialIO (F.softmax (Dim (-1)) lastLogits) 1 True
      loop (F.cat (Dim 1) [current, next]) (n - 1)
