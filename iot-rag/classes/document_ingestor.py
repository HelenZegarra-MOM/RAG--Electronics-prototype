import os
import logging
from pathlib import Path
from classes.utilities import clean_text_from_file

logger = logging.getLogger(__name__)

class DocumentIngestor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_files(self):
        files = [f for f in os.listdir(self.input_dir) if f.endswith(".txt")]

        for filename in files:
            input_path = self.input_dir / filename
            cleaned_text = clean_text_from_file(input_path)

            output_path = self.output_dir / filename
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            logger.info(f"🧹 Cleaned and saved: {filename}")

        logger.info("✅ Document ingestion completed.")
