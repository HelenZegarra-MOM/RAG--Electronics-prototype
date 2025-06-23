import os
import shutil
import logging

logger = logging.getLogger(__name__)

def clean_text_from_file(filepath: str) -> str:
    """Reads and cleans text from a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to clean text from {filepath}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Splits text into chunks of fixed size with optional overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def delete_directory(dir_path: str):
    """Deletes a directory and its contents."""
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            logger.info(f"Deleted directory: {dir_path}")
        else:
            logger.warning(f"Directory not found for deletion: {dir_path}")
    except Exception as e:
        logger.error(f"Error deleting directory {dir_path}: {e}")


def ensure_clean_dir(path):
    import shutil
    import os
    import logging
    logger = logging.getLogger(__name__)
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            logger.info(f"Deleted existing directory: {path}")
        except PermissionError as e:
            logger.error(f"Permission denied deleting directory {path}: {e}")
            return  # optionally: raise or skip
    os.makedirs(path, exist_ok=True)
