{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

-- | A small decoder-only language model for reading and experimenting on CPU.
--
-- = Reading path
--
-- Begin with "NanoChat.Tokenizer", then @sampleBatch@ in "NanoChat.Training".
-- Follow 'modelForwardWith' through embeddings, 'blockForward', and
-- 'headForward'. Return to @trainStep@ for the Adam update, then read 'generate'
-- and "NanoChat.Checkpoint" for inference and persistence.
--
-- = Shapes
--
-- @B@ = batch size, @T@ = sequence length, @D@ = embedding width,
-- @H@ = head count, @K = D / H@ = head width, @V@ = vocabulary size.
--
-- @
-- IDs [B,T] -> embeddings [B,T,D] -> blocks [B,T,D] -> logits [B,T,V]
-- one head: Q,K,V [B,T,K] -> scores [B,T,T] -> output [B,T,K]
-- @
--
-- This uses Hasktorch's untyped API: these dimensions are runtime contracts,
-- not guarantees in the Haskell types. The model and its @Config@ must agree.
-- All tensors in this implementation are on CPU.
--
-- = Relation to the original Transformer
--
-- Scaled dot-product attention follows equation 1 of
-- <https://arxiv.org/abs/1706.03762 Vaswani et al., Attention Is All You Need (2017)>.
-- This example has no encoder or cross-attention. It normalizes /before/ each
-- sublayer and uses learned positions; it is not the paper's full architecture
-- or a checkpoint-compatible implementation of Karpathy's NanoChat.
--
-- = Haskell and Hasktorch
--
-- @Randomizable spec model@ supplies @sample :: spec -> IO model@: the spec
-- describes dimensions, while the result holds initialized parameters.
-- @Parameterized@ supplies traversal of those parameters; the @Generic@
-- instances let Hasktorch traverse our nested records in a consistent order.
-- @flattenParameters@ collects them for Adam and checkpoint serialization.
-- @makeIndependent@ turns an initialized tensor into a trainable leaf parameter;
-- @toDependent@ exposes that parameter as a tensor used by the forward graph.
--
-- 'modelForwardWith' takes a dropout action. Evaluation supplies @Identity@,
-- which leaves values unchanged; training supplies an @IO@ action that randomly
-- drops activations. Both therefore share the same attention and loss code.

module NanoChat.Transformer where

import Control.Monad (foldM, replicateM)
import Data.Functor.Identity (Identity (..))
import Data.Aeson (FromJSON, ToJSON)
import GHC.Generics (Generic)
import Torch (Tensor, Parameter, Dim (..), KeepDim (..))
import qualified Torch as T
import qualified Torch.Functional as F
import qualified Torch.NN as NN

-- | Architecture dimensions and dropout probability. Batch size, learning rate,
-- and training step count belong to the training procedure, not this record.
data Config = Config
  { blockSize :: Int, -- ^ Maximum number of input tokens per sequence.
  nEmbd :: Int, -- ^ Embedding width @D@; divisible by head count.
  nHead :: Int, -- ^ Parallel attention heads @H@ in each block.
  nLayer :: Int, -- ^ Number of transformer blocks.
  vocabSize :: Int, -- ^ Vocabulary entries @V@, including reserved tokens.
  dropout :: Double -- ^ Training-only dropout probability in @[0,1)@.
  } deriving (Show, Eq, Generic)

instance ToJSON Config
instance FromJSON Config

-- | Default architecture: context 128, width 128, four heads, three blocks,
-- dropout 0.1. The trainer replaces the placeholder vocabulary size with the
-- size of the tokenizer built from training text.
defaultConfig :: Config
defaultConfig = Config 128 128 4 3 256 0.1

-- | Small CPU preset used by the @--tiny@ learning example.
-- Context 16, width 32, two heads, one block, dropout 0.1. The vocabulary size
-- is a placeholder until the training tokenizer has been built.
tinyConfig :: Config
tinyConfig = Config
  { blockSize = 16, nEmbd = 32, nHead = 2, nLayer = 1
  , vocabSize = 256, dropout = 0.1 }

-- | Check positive dimensions, divisibility of width by head count, and dropout.
-- Does not check that an already-created model has these dimensions.
validateConfig :: Config -> Either String ()
validateConfig Config{..}
  | any (<= 0) [blockSize, nEmbd, nHead, nLayer, vocabSize] =
      Left "Model dimensions must be positive."
  | nEmbd `mod` nHead /= 0 = Left "nEmbd must be divisible by nHead."
  | isNaN dropout || dropout < 0 || dropout >= 1 = Left "dropout must be in [0, 1)."
  | otherwise = Right ()

-- | Dimensions needed to initialize the query, key, and value projections.
data HeadSpec = HeadSpec { headSize :: Int, config :: Config }
-- | One attention head, with three distinct learned projections @D -> K@.
data Head = Head
  { wQuery :: NN.Linear, 
  wKey :: NN.Linear, 
  wValue :: NN.Linear } deriving (Generic)

-- Generic traversal finds the parameters inside all three linear layers.
-- Sampling runs the initialization action three times, creating distinct weights.
instance NN.Parameterized Head
instance NN.Randomizable HeadSpec Head where
  sample (HeadSpec width cfg) = Head <$> linear <*> linear <*> linear
    where linear = NN.sample (NN.LinearSpec (nEmbd cfg) width)

-- | Parallel heads followed by a learned @D -> D@ projection.
-- Concatenating @H@ head outputs of width @K@ restores width @D = H*K@.
data MultiHeadAttention = MultiHeadAttention
  { heads :: [Head], 
    proj :: NN.Linear } deriving (Generic)
instance NN.Parameterized MultiHeadAttention

-- | Position-wise network: @D -> 4*D -> D@, with ReLU between the layers.
-- The same weights are applied independently at each batch/time position.
data FeedForward = FeedForward
  { fc1 :: NN.Linear, 
    fc2 :: NN.Linear } deriving (Generic)
instance NN.Parameterized FeedForward

-- | A trainable lookup table, stored as a parameter rather than a plain tensor.
-- Token embeddings have shape @[V,D]@; position embeddings @[blockSize,D]@.
newtype Embedding = Embedding Parameter deriving (Generic)
instance NN.Parameterized Embedding

-- | Look up embedding rows. Int64 IDs @[B,T]@ produce @[B,T,D]@;
-- position IDs @[T]@ produce @[T,D]@. IDs must be within the table's range.
embeddingForward :: Embedding -> Tensor -> Tensor
embeddingForward (Embedding weights) = F.embedding' (T.toDependent weights)

-- | Learned gain and bias, each @[D]@, for normalization over the final axis.
data LayerNorm = LayerNorm Parameter Parameter deriving (Generic)
instance NN.Parameterized LayerNorm

-- | Initialize gain to ones and bias to zeros, with the supplied width.
-- Despite its name, this initialization itself uses no random values.
sampleLayerNorm :: Int -> IO LayerNorm
sampleLayerNorm width = LayerNorm
  <$> T.makeIndependent (T.ones' [width])
  <*> T.makeIndependent (T.zeros' [width])

-- | Normalize each token's embedding components, preserving shape @[B,T,D]@.
-- Mean and variance are taken over @D@, never over batch or time. Dividing by
-- @sqrt (variance + 1e-5)@ avoids division by zero; learned gain and bias follow.
layerNormForward :: LayerNorm -> Tensor -> Tensor
layerNormForward (LayerNorm gain bias) x =
  let meanLast = F.meanDim (Dim (-1)) KeepDim (T.dtype x)
      centered = x - meanLast x
      variance = meanLast (centered * centered)
  in centered / F.sqrt (variance + 1e-5) * T.toDependent gain + T.toDependent bias

-- | A pre-normalized attention sublayer and feed-forward sublayer, each with
-- a residual connection. Both input and output have shape @[B,T,D]@.
data Block = Block
  { sa :: MultiHeadAttention, 
    ffwd :: FeedForward,
    ln1 :: LayerNorm, 
    ln2 :: LayerNorm } deriving (Generic)
instance NN.Parameterized Block

-- | Parameters of the complete decoder-only language model.
-- The final linear layer maps each hidden vector to @V@ unnormalized scores
-- (logits). Its weights are separate from the input embedding table.
data NanoChatModel = NanoChatModel
  { tokenEmbedding :: Embedding, 
    positionEmbedding :: Embedding, 
    blocks :: [Block], 
    lnFinal :: LayerNorm, 
    lmHead :: NN.Linear
    } deriving (Generic)
instance NN.Parameterized NanoChatModel

-- | Initialization recipe used by @T.sample@. Embeddings start from normal
-- random values scaled by 0.02; linear layers use Hasktorch's initializer.
-- This creates an untrained model, not a sample of text or a training batch.
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

-- | Evaluate with dropout disabled. Inputs are Int64 IDs @[B,T]@, optional
-- targets also @[B,T]@. Returns logits @[B,T,V]@ and, when targets are supplied,
-- a scalar mean next-token cross-entropy. See 'modelForwardWith' for contracts.
modelForward :: NanoChatModel -> Config -> Tensor -> Maybe Tensor -> (Tensor, Maybe Tensor)
modelForward model cfg idx targets =
  runIdentity (modelForwardWith Identity model cfg idx targets)

-- | Forward pass with training dropout enabled; shapes and loss are the same
-- as 'modelForward'. This computes predictions and loss but does not update
-- weights; Adam is called separately by @trainStep@ in "NanoChat.Training".
modelForwardTrain :: NanoChatModel -> Config -> Tensor -> Maybe Tensor -> IO (Tensor, Maybe Tensor)
modelForwardTrain model cfg = modelForwardWith (F.dropout (dropout cfg) True) model cfg

-- | Shared forward pass with a caller-supplied dropout action.
--
-- Token IDs @[B,T]@ index token embeddings. Position embeddings @[T,D]@ are
-- broadcast across the batch, giving @[B,T,D]@. Each block preserves this
-- shape. Final normalization and a linear projection produce logits @[B,T,V]@.
--
-- Targets must have the same shape as inputs and must already be shifted one
-- token forward. @logSoftmax@ followed by @nllLoss'@ computes mean
-- cross-entropy across all @B*T@ positions, in natural-log units per token.
-- With no targets, the loss is 'Nothing'. No assistant-only loss mask is used.
--
-- Requires a matching model/configuration, @B > 0@, @1 <= T <= blockSize@,
-- and IDs in @[0,V)@. Invalid input rank/length or target shape raises an error.
-- Other dimension and ID errors may be reported by LibTorch.
modelForwardWith :: Monad m => (Tensor -> m Tensor) -> NanoChatModel -> Config
                 -> Tensor -> Maybe Tensor -> m (Tensor, Maybe Tensor)
modelForwardWith dropTensor NanoChatModel{..} Config{..} idx targets
  | length (T.shape idx) /= 2 = error "Input must have shape [batch, time]."
  | t < 1 || t > blockSize = error "Input length must be between 1 and blockSize."
  | otherwise = do
      let positions = T.asTensor ([0 .. t - 1] :: [Int])
      x <- dropTensor (embeddingForward tokenEmbedding idx + embeddingForward positionEmbedding positions)
      hidden <- foldM (blockForward dropTensor) x blocks
      let logits = NN.linearForward lmHead (layerNormForward lnFinal hidden)
          loss y
            | T.shape y /= T.shape idx = error "Targets must have the same shape as inputs."
            | otherwise = F.nllLoss' (T.reshape [b * t] y)
                (F.logSoftmax (Dim 1) (T.reshape [b * t, vocabSize] logits))
      pure (logits, fmap loss targets)
  where
    b = T.size 0 idx
    t = T.size 1 idx

-- | Run one causal attention head: @[B,T,D] -> [B,T,K]@.
-- Queries, keys, and values have shape @[B,T,K]@. Multiplying queries by
-- transposed keys gives scores @[B,T,T]@, scaled by @1 / sqrt K@.
-- A causal mask permits the current position and earlier positions; future
-- scores become negative infinity before softmax, giving them zero weight.
-- Softmax normalizes over key positions (the last axis), then dropout is applied.
-- Multiplication by the values produces the head's output.
-- See equation 1 and section 3.2.3 of
-- <https://arxiv.org/abs/1706.03762 Attention Is All You Need>.
headForward :: Monad m => (Tensor -> m Tensor) -> Head -> Tensor -> m Tensor
headForward dropTensor Head{..} x = do
  let t = T.size 1 x
      q = NN.linearForward wQuery x
      k = NN.linearForward wKey x
      v = NN.linearForward wValue x
      scale = 1 / sqrt (fromIntegral (T.size (-1) k) :: Float)
      scores = F.mulScalar scale (F.matmul q (F.transpose (Dim (-2)) (Dim (-1)) k))
      -- Additive [time,time] mask broadcasts across the batch.
      mask = T.asTensor
      -- j > i is the future; the diagonal j == i remains visible.
        [[if j > i then negate (1 / 0) else 0 :: Float | j <- [0 .. t - 1]]
        | i <- [0 .. t - 1]]
      masked = scores + mask
  weights <- dropTensor (F.softmax (Dim (-1)) masked)
  pure (F.matmul weights v)

-- | Preserve shape @[B,T,D]@ through attention and feed-forward sublayers.
-- Normalize once for all heads, concatenate their outputs, and project back
-- to width @D@. Add the original input as a residual. A second normalization
-- feeds the @D -> 4*D -> D@ network, followed by another residual addition.
-- The dropout action is shared with 'headForward'; each call draws its own
-- training mask. The feed-forward form follows section 3.3 of
-- <https://arxiv.org/abs/1706.03762 Attention Is All You Need>, but our
-- normalization is /before/ the sublayers.
blockForward :: Monad m => (Tensor -> m Tensor) -> Tensor -> Block -> m Tensor
blockForward dropTensor x Block{..} = do
  let normalized = layerNormForward ln1 x
  hs <- mapM (\h -> headForward dropTensor h normalized) (heads sa)
  attention <- dropTensor (NN.linearForward (proj sa) (F.cat (Dim (-1)) hs))
  let residual = x + attention
      FeedForward{..} = ffwd
      feed = NN.linearForward fc2 (F.relu (NN.linearForward fc1 (layerNormForward ln2 residual)))
  output <- dropTensor feed
  pure (residual + output)



--------- Generate ---

-- | Append a requested number of sampled token IDs: @[B,T] -> [B,T+count]@.
-- Requires a nonempty Int64 prompt of valid IDs and a matching configuration.
-- Each step evaluates only the last 'blockSize' tokens with dropout disabled,
-- samples from the final position's softmax, and appends one ID per sequence.
-- Positions restart at zero within each cropped context.
--
-- Negative counts raise an IO error. A zero count returns the prompt unchanged.
-- Sampling is random; no temperature adjustment or top-k filtering is used.
-- This function does not stop at @[END]@; the chat executable hides text after
-- the first generated end marker and starts a fresh prompt for every user turn.
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
