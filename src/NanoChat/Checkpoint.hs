{-# LANGUAGE DeriveGeneric #-}

-- | A single, versioned model file: JSON metadata followed by parameter tensors.
module NanoChat.Checkpoint
  ( ModelInfo (..), saveModel, loadModel, ensureNewModelFile ) where

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
import NanoChat
import NanoChat.Tokenizer

data ModelInfo = ModelInfo
  { formatVersion :: Int
  , modelConfig :: Config
  , vocabulary :: [Text.Text]
  , completedSteps :: Int
  , datasetPath :: FilePath
  , finalTrainingLoss :: Float
  , trainingBatchSize :: Int
  , trainingLearningRate :: Float
  } deriving (Show, Eq, Generic)

instance ToJSON ModelInfo
instance FromJSON ModelInfo

ensureNewModelFile :: FilePath -> IO ()
ensureNewModelFile path = do
  exists <- doesPathExist path
  when exists $ ioError (userError ("Model file already exists: " ++ path ++ ". Choose a new name."))

validateInfo :: ModelInfo -> IO Tokenizer
validateInfo info = do
  unless (formatVersion info == 1) $ ioError (userError "Unsupported model file version.")
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

saveModel :: FilePath -> ModelInfo -> NanoChatModel -> IO ()
saveModel path info model = do
  _ <- validateInfo info
  ensureNewModelFile path
  createDirectoryIfMissing True (takeDirectory path)
  -- A byte tensor keeps UTF-8 JSON and weights together in one LibTorch archive.
  let metadata = T.asTensor (Bytes.unpack (encode info))
      tensors = metadata : map T.toDependent (T.flattenParameters model)
  -- Publish only after the archive is fully written; a failed save leaves no
  -- partially written model at the chosen name.
  bracketOnError
    (do (temporary, handle) <- openBinaryTempFile (takeDirectory path) (takeFileName path ++ ".tmp")
        hClose handle
        pure temporary)
    removeFile
    (\temporary -> do
      Serialize.save tensors temporary
      ensureNewModelFile path
      renameFile temporary path)

loadModel :: FilePath -> IO (NanoChatModel, Tokenizer, ModelInfo)
loadModel path = do
  tensors <- Serialize.load path
  case tensors of
    [] -> ioError (userError "Empty model archive.")
    metadata : weights -> do
      unless (T.dtype metadata == T.UInt8 && length (T.shape metadata) == 1) $
        ioError (userError "Invalid model metadata tensor.")
      info <- either (ioError . userError . ("Invalid model metadata: " ++)) pure
        (eitherDecode (Bytes.pack (T.asValue metadata :: [Word8])))
      tokenizer <- validateInfo info
      template <- T.sample (NanoChatModelSpec (modelConfig info))
      let expected = map T.toDependent (T.flattenParameters template)
          compatible a b = T.shape a == T.shape b && T.dtype a == T.dtype b
      unless (length weights == length expected && and (zipWith compatible weights expected)) $
        ioError (userError "Saved parameter count, shape or dtype does not match model configuration.")
      parameters <- mapM T.makeIndependent weights
      pure (T.replaceParameters template parameters, tokenizer, info)
