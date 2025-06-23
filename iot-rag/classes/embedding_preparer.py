import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class EmbeddingPreparer:
    def __init__(self,
                 file_list,
                 input_dir,
                 output_dir,
                 embedding_model_name):
        self.file_list = file_list or []
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.embedding_model_name = embedding_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.model = AutoModel.from_pretrained(self.embedding_model_name).to(self.device)
            self.logger.info(f"Loaded model/tokenizer: {self.embedding_model_name} on {self.device}")
        except Exception as e:
            self.logger.critical(f"Failed to load model or tokenizer: {e}")
            raise

    def process_files(self):
        original_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.INFO)

        for filename in self.file_list:
            if not filename:
                self.logger.warning("Empty filename encountered, skipping.")
                continue

            file_path = self.input_dir / filename
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}, skipping.")
                continue

            try:
                self.logger.info(f"Processing file: {file_path}")
                text = self._read_file(file_path)
                if not text:
                    self.logger.warning(f"No text found in {file_path}, skipping.")
                    continue

                embedding = self._generate_embedding(text)
                if embedding is not None:
                    self._save_embedding(file_path, embedding, text)
                else:
                    self.logger.warning(f"Failed to generate embedding for {file_path}")

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")

        logging.getLogger().setLevel(original_level)

    def _read_file(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return None

    def _generate_embedding(self, text):
        if not text:
            self.logger.warning("Empty text provided for embedding generation.")
            return None

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Mean pooling over token embeddings
            embedding_tensor = outputs.last_hidden_state.mean(dim=1).squeeze()
            embedding = embedding_tensor.cpu().numpy()
            return embedding.tolist()  # Convert numpy array to list for JSON serialization
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return None

    def _save_embedding(self, file_path, embedding, text):
        output_file = self.output_dir / f"{file_path.stem}_embedding.json"
        metadata_file = self.output_dir / f"{file_path.stem}_metadata.json"

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(embedding, f)

            with open(metadata_file, "w", encoding="utf-8") as mf:
                json.dump({
                    "source_file": file_path.name,
                    "text_preview": text[:300]
                }, mf)

            self.logger.info(f"Saved embedding and metadata for {file_path.name}")
        except Exception as e:
            self.logger.error(f"Error saving embedding or metadata for {file_path}: {e}")
