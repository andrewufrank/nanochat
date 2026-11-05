{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad (foldM, when)
import Data.List (foldl')
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import System.IO (hFlush, stdout)

-- Externe Haskell-Bibliotheken für Vektor-Berechnungen (TorchHaskell / HaskTorch)
import Torch
import qualified Torch.Functional as F
import qualified Torch.NN as NN
import qualified Torch.Optim as Optim

-- Importiere die Typen und Funktionen aus deinem NanoChat-Modul
import NanoChat

-- =====================================================================
-- 1. EIN EINFACHER CHAR- / SUBWORD-TOKENIZER FÜR HASKELL
-- =====================================================================
-- Um Abhängigkeiten flach zu halten, mappen wir hier Text-Tokens auf Int-IDs.
-- Im echten Einsatz würde man hier via FFI BPE (Byte-Pair-Encoding) anbinden.

data Tokenizer = Tokenizer
  { tokenToId :: T.Text -> Int
  , idToToken :: Int -> T.Text
  }

-- Erstellt ein Vokabular aus dem gesamten Text
buildTokenizer :: T.Text -> Tokenizer
buildTokenizer rawText =
  let tokens = T.words rawText ++ ["[USER]", "[BOT]", "[END]"]
      uniqueTokens = unique tokens
      vocab = zip uniqueTokens [0..]
      forwardMap = \t -> maybe 0 id (lookup t vocab)
      backwardMap = \i -> if i < length uniqueTokens then uniqueTokens !! i else "[UNK]"
  in Tokenizer forwardMap backwardMap
  where
    unique = foldl' (\acc x -> if x `elem` acc then acc else acc ++ [x]) []

-- =====================================================================
-- 2. DATA LOADER: BATCHES ZUFÄLLIG GENERIEREN
-- =====================================================================
-- Generiert ein Tuple aus Input (X) und Target (Y) Tensoren
getBatch :: Tensor -> Int -> Int -> IO (Tensor, Tensor)
getBatch allTokens bSize bkSize = do
  let totalTokens = size allTokens 0
  -- Zufällige Start-Indizes für das CPU-Training würfeln
  ix <- F.randint 0 (totalTokens - bkSize - 1) [bSize] (withDevice CPU)
  let ixList = asValue ix :: [Int]
      
      -- Slices extrahieren für X und Y (Targets sind um 1 verschoben)
      extractX i = F.slice 0 i (i + bkSize) 1 allTokens
      extractY i = F.slice 0 i (i + bkSize + 1) 1 allTokens
      
      xBatch = F.stack (Dim 0) (map extractX ixList)
      yBatch = F.stack (Dim 0) (map extractY ixList)
  
  return (xBatch, yBatch)

-- =====================================================================
-- 3. INTERAKTIVE CHAT-SCHLEIFE (NACH DEM TRAINING)
-- =====================================================================
chatLoop :: NanoChatModel -> Config -> Tokenizer -> IO ()
chatLoop model cfg tokenizer = do
  putStr "\n[USER]: "
  hFlush stdout
  input <- TIO.getLine
  when (not (T.null input)) $ do
    let prompt = "[USER] " <> input <> " [BOT]"
        -- Konvertiere Eingabe in Token-IDs
        inputTokens = map (tokenToId tokenizer) (T.words prompt)
        contextIdx  = F.asTensor inputTokens
        -- Füge Batch-Dimension hinzu (1, T)
        contextBatch = F.unsqueeze (Dim 0) contextIdx 
    
    -- Modell generiert bis zu 30 neue Tokens
    responseIdx <- generate model cfg contextBatch 30
    
    -- Konvertiere IDs zurück in lesbaren Text
    let generatedIds = asValue (F.squeeze responseIdx) :: [Int]
        outputText   = T.unwords (map (idToToken tokenizer) generatedIds)
        
    TIO.putStrLn $ "[BOT-OUTPUT]: " <> outputText
    chatLoop model cfg tokenizer

-- =====================================================================
-- 4. HAUPTPROGRAMM (MAIN)
-- =====================================================================
main :: IO ()
main = do
  putStrLn "=== Starte Haskell NanoChat CPU-Training ==="
  
  -- 1. Datensatz einlesen (Erwartet z.B. eine reine Textdatei mit deinen Dialogen)
  putStrLn "Lese Datensatz 'chat_dataset.txt' ein..."
  rawContent <- TIO.readFile "chat_dataset.txt"
  
  -- 2. Tokenizer aufbauen und Text numerisch kodieren
  let tokenizer = buildTokenizer rawContent
      encodedText = map (tokenToId tokenizer) (T.words rawContent)
      allTokensTensor = F.asTensor encodedText
      
      -- Dynamische Anpassung der Vokabular-Größe an unsere Daten
      cfg = defaultConfig { vocabSize = length encodedText }
  
  putStrLn $ "Datensatz erfolgreich geladen. Tokens: " ++ show (length encodedText)
  
  -- 3. Modell und mathematischen Optimizer initialisieren
  -- HaskTorch/TorchHaskell nutzt 'sample' zur zufälligen Gewichts-Initialisierung
  initModel <- NN.sample (NanoChatModelSpec cfg) 
  let initOptimizer = Optim.adam 0.001 (NN.flattenParameters initModel)
  
  -- 4. Trainingsschleife (500 Schritte auf der CPU)
  putStrLn "Starte CPU-Optimierung (500 Iterationen)..."
  (trainedModel, _) <- foldM (trainStep allTokensTensor cfg) (initModel, initOptimizer) [1..500]
  
  putStrLn "\nTraining erfolgreich beendet!"
  
  -- 5. In den Live-Chat-Modus wechseln
  putStrLn "Du kannst jetzt mit deinem Modell chatten (Leerzeile zum Beenden):"
  chatLoop trainedModel cfg tokenizer

-- Ein einzelner, reiner Trainingsschritt (Forward -> Backward -> Step)
trainStep :: Tensor -> Config -> (NanoChatModel, Optim.Adam) -> Int -> IO (NanoChatModel, Optim.Adam)
trainStep allTokens cfg (model, optimizer) step = do
  -- Batch holen (BatchSize = 8, BlockSize = 128)
  (xb, yb) <- getBatch allTokens 8 128
  
  -- Forward Pass
  let (_, mLoss) = modelForward model cfg xb (Just yb)
  
  case mLoss of
    Nothing -> return (model, optimizer)
    Just loss -> do
      -- Konsolen-Output alle 50 Schritte
      when (step `mod` 50 == 0) $
        putStrLn $ "Schritt " ++ show step ++ " | CPU Loss: " ++ show (asValue loss :: Float)
      
      -- Backward Pass (Berechnung der Gradienten)
      let gradients = backward loss
      -- Gewichte über den Optimizer anpassen und aktualisierten Zustand zurückgeben
      let (nextModel, nextOptimizer) = Optim.step optimizer gradients model
      return (nextModel, nextOptimizer)
