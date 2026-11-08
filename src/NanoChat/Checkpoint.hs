{-# LANGUAGE DeriveGeneric #-}

-- | Save and restore the state needed for chat or continued training.
--
-- Version 1 stores JSON metadata and model parameters. Version 2 adds Adam's
-- scalar settings and two moment tensors per parameter. Tensor order follows
-- @flattenParameters@; count, shape, dtype, and device are checked on loading.
-- The exact vocabulary order and architecture are restored from metadata.
--
-- Start with "NanoChat.Training" to understand the model/optimizer state being
-- saved. RNG state is not saved, so resumed random batches and dropout need not
-- match an uninterrupted run. Files are written only when a run finishes.
module NanoChat.Checkpoint
  ( ModelInfo (..), saveModel, loadModel, ensureNewModelFile
  , saveTrainingCheckpoint, loadTrainingCheckpoint
  ) where

import Control.Exception (bracketOnError)
import Control.Monad (unless, when)
import Data.Aeson (FromJSON, ToJSON, eitherDecode, encode)
import qualified Data.ByteString.Lazy as Bytes
import qualified Data.Text as Text
import Data.Word (Word8)
import GHC.Generics (Generic)
import System.Directory (createDirectoryIfMissing, doesPathExist, removeFile, renameFile)
import System.FilePath (takeDirectory, takeFileName)
import System.IO (hClose, openBinaryTempFile)
import qualified Torch as T
import qualified Torch.Serialize as Serialize
import qualified Torch.Optim as Optim
import NanoChat.Transformer
import NanoChat.Tokenizer

-- | Metadata describing the saved model and the last training run.
-- Loss is the last /training batch/ loss before its update, not validation loss.
-- Validation scores are printed separately by the training executable.
data ModelInfo = ModelInfo
  { formatVersion :: Int -- ^ 1 for model-only; 2 for model plus Adam.
  , modelConfig :: Config -- ^ Architecture used to reconstruct the model.
  , vocabulary :: [Text.Text] -- ^ Exact words in token-ID order.
  , completedSteps :: Int -- ^ Cumulative updates; equals the saved Adam counter.
  , datasetPath :: FilePath -- ^ Most recent training file; its contents are not saved.
  , finalTrainingLoss :: Float -- ^ Last batch mean loss, in nats per target token.
  , trainingBatchSize :: Int -- ^ Sequences per training batch.
  , trainingLearningRate :: Float -- ^ Adam learning rate restored on resume.
  } deriving (Show, Eq, Generic)

instance ToJSON ModelInfo
instance FromJSON ModelInfo

-- | Refuse an existing output path with an IO error. Training calls this before
-- doing expensive work; saving checks it again before publishing the archive.
ensureNewModelFile :: FilePath -> IO ()
ensureNewModelFile path = do
  exists <- doesPathExist path
  when exists $ ioError (userError ("Model file already exists: " ++ path ++ ". Choose a new name."))

validateInfo :: ModelInfo -> IO Tokenizer
validateInfo info = do
  unless (formatVersion info `elem` [1, 2]) $ ioError (userError "Unsupported model file version.")
  either (ioError . userError) pure (validateConfig (modelConfig info))
  tokenizer <- either (ioError . userError) pure (tokenizerFromVocabulary (vocabulary info))
  unless (tokenizerSize tokenizer == vocabSize (modelConfig info)) $
    ioError (userError "Saved vocabulary size does not match model configuration.")
  when (completedSteps info < 0 || trainingBatchSize info <= 0 ||
        not (finite (trainingLearningRate info)) || trainingLearningRate info <= 0 ||
        not (finite (finalTrainingLoss info))) $
    ioError (userError "Invalid training metadata.")
  pure tokenizer
  where finite value = not (isNaN value || isInfinite value)

-- | Save a version-1 model-only archive. Requires @formatVersion == 1@ and
-- valid metadata. Existing output paths are refused. New training runs use
-- 'saveTrainingCheckpoint' so they retain Adam history.
saveModel :: FilePath -> ModelInfo -> NanoChatModel -> IO ()
saveModel path info model = do
  _ <- validateInfo info
  unless (formatVersion info == 1) $
    ioError (userError "Use saveTrainingCheckpoint for version-2 archives.")
  writeArchive path (jsonTensor info : modelWeights model)

-- Scalar optimizer state is stored as JSON; moment tensors retain their precision.
data AdamInfo = AdamInfo
  { adamBeta1 :: Float, adamBeta2 :: Float, adamSteps :: Int }
  deriving (Generic)
instance ToJSON AdamInfo
instance FromJSON AdamInfo

-- | Save model weights, metadata, and Adam state in a version-2 archive.
-- Requires @formatVersion == 2@, matching step counts and compatible moments.
-- Writes a temporary file before publishing the completed archive; existing
-- output paths are refused. Does not save RNG state or the dataset itself.
saveTrainingCheckpoint :: FilePath -> ModelInfo -> NanoChatModel -> Optim.Adam -> IO ()
saveTrainingCheckpoint path info model optimizer = do
  _ <- validateInfo info
  unless (formatVersion info == 2) $
    ioError (userError "Training checkpoints require format version 2.")
  validateAdam info (modelWeights model) optimizer
  let adam = AdamInfo (Optim.beta1 optimizer) (Optim.beta2 optimizer) (Optim.iter optimizer)
  writeArchive path (jsonTensor info : jsonTensor adam :
    (modelWeights model ++ Optim.m1 optimizer ++ Optim.m2 optimizer))

modelWeights :: NanoChatModel -> [T.Tensor]
modelWeights = map T.toDependent . T.flattenParameters

jsonTensor :: ToJSON a => a -> T.Tensor
jsonTensor = T.asTensor . Bytes.unpack . encode

readJSONTensor :: FromJSON a => T.Tensor -> IO a
readJSONTensor tensor = do
  unless (T.dtype tensor == T.UInt8 && length (T.shape tensor) == 1) $
    ioError (userError "Invalid metadata tensor.")
  either (ioError . userError . ("Invalid checkpoint metadata: " ++)) pure
    (eitherDecode (Bytes.pack (T.asValue tensor :: [Word8])))

writeArchive :: FilePath -> [T.Tensor] -> IO ()
writeArchive path tensors = do
  ensureNewModelFile path
  createDirectoryIfMissing True (takeDirectory path)
  -- Publish only after the archive is fully written.
  bracketOnError
    (do (temporary, handle) <- openBinaryTempFile (takeDirectory path) (takeFileName path ++ ".tmp")
        hClose handle
        pure temporary)
    removeFile
    (\temporary -> do
      Serialize.save tensors temporary
      ensureNewModelFile path
      renameFile temporary path)

-- | Load either version for inference, returning model, tokenizer, and metadata.
-- Restores the saved configuration rather than the current default. Malformed
-- metadata or incompatible tensors raise an IO error. Version-2 Adam state is
-- validated but omitted from the result. No original dataset is needed.
loadModel :: FilePath -> IO (NanoChatModel, Tokenizer, ModelInfo)
loadModel path = do
  (model, tokenizer, info, _) <- loadArchive path
  pure (model, tokenizer, info)

-- | Load a version-2 model and its Adam state for continued training.
-- Returns model, tokenizer, metadata, and optimizer. Version-1 files raise an
-- IO error because they contain no Adam history; no fresh optimizer is silently
-- substituted. Callers encode new text using the returned tokenizer.
loadTrainingCheckpoint :: FilePath -> IO (NanoChatModel, Tokenizer, ModelInfo, Optim.Adam)
loadTrainingCheckpoint path = do
  (model, tokenizer, info, optimizer) <- loadArchive path
  case optimizer of
    Nothing -> ioError (userError "This version-1 model has no Adam state and cannot resume training. It remains usable for chat.")
    Just adam -> pure (model, tokenizer, info, adam)

loadArchive :: FilePath -> IO (NanoChatModel, Tokenizer, ModelInfo, Maybe Optim.Adam)
loadArchive path = do
  tensors <- Serialize.load path
  case tensors of
    [] -> ioError (userError "Empty model archive.")
    metadata : payload -> do
      info <- readJSONTensor metadata
      tokenizer <- validateInfo info
      template <- T.sample (NanoChatModelSpec (modelConfig info))
      let expected = modelWeights template
          count = length expected
      (weights, optimizer) <- case formatVersion info of
        1 -> do
          validateTensors "model parameters" expected payload
          pure (payload, Nothing)
        _ -> case payload of
          [] -> ioError (userError "Missing Adam metadata.")
          adamMetadata : values -> do
            adam <- readJSONTensor adamMetadata
            let (weights, moments) = splitAt count values
                (firstMoments, secondMoments) = splitAt count moments
                optimizer = Optim.Adam (adamBeta1 adam) (adamBeta2 adam)
                  firstMoments secondMoments (adamSteps adam)
            validateTensors "model parameters" expected weights
            validateAdam info expected optimizer
            pure (weights, Just optimizer)
      parameters <- mapM T.makeIndependent weights
      pure (T.replaceParameters template parameters, tokenizer, info, optimizer)

validateTensors :: String -> [T.Tensor] -> [T.Tensor] -> IO ()
validateTensors label expected actual =
  unless (length actual == length expected && and (zipWith compatible expected actual)) $
    ioError (userError ("Saved " ++ label ++ " count, shape, dtype or device does not match model configuration."))
  where
    compatible a b = T.shape a == T.shape b && T.dtype a == T.dtype b && T.device a == T.device b

validateAdam :: ModelInfo -> [T.Tensor] -> Optim.Adam -> IO ()
validateAdam info expected optimizer = do
  let validBeta value = not (isNaN value || isInfinite value) && value >= 0 && value < 1
  unless (validBeta (Optim.beta1 optimizer) && validBeta (Optim.beta2 optimizer) &&
          Optim.iter optimizer == completedSteps info && Optim.iter optimizer >= 0) $
    ioError (userError "Invalid Adam settings or step count does not match training metadata.")
  validateTensors "Adam first moments" expected (Optim.m1 optimizer)
  validateTensors "Adam second moments" expected (Optim.m2 optimizer)
