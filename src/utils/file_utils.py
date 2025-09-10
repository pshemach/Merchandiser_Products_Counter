import os
import json
import logging
import pickle
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

def ensure_dir(directory:Path) -> Path:
    """Ensure directory exists, create if it does not"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensures directory exists: {directory}")
    return directory


def load_json(filepath:Path) -> Dict[str, Any]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(F"Loaded JSON from file: {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"JSON file not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file: {filepath}")
        raise
    
def save_json(data: Dict[str, Any], filepath: Path, indent: int=2) -> None:
    """Save data to JSON file"""
    try:
        ensure_dir(Path(filepath).parent)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.debug(f"Saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        raise 
    
def save_pickle(data: Any, filepath: Path) -> None:
    try:
        ensure_dir(Path(filepath).parent)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f) 
        logger.debug(f"Saved pickle to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save pickle to {filepath}: {e}")
        raise
    
def load_pickle(filepath: Path) -> Any:
    try:
        # ensure_dir(Path(filepath).parent)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logger.debug(f"Loaded pickle file from {filepath}")   
        return data
    except Exception as e:
        logger.error(f"Failed to loaded pickle file from {filepath}: {e}")
        raise

def get_file_size(filepath: Path) -> int:
    """Get file size in bytes"""
    return Path(filepath).stat().st_size

def list_images(directory: Path, extensions: List[str]=None) -> List[Path]:
    """list all images files in directory"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory does not exists: {directory}")
        return []
    
    images = []
    for ext in extensions:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))
        
    logger.info(f"Found {len(images)} images in {directory}")
    return sorted(images)