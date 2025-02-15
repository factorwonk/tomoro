import json
from typing import List, Dict, Any
from pathlib import Path

def load_json_data(path: str, sample_size: int = 10) -> List[Dict[str, Any]]:
    """
    Load and optionally sample JSON data from a specified file path.
    This function handles file reading, JSON parsing, and data sampling
    with appropriate error handling.
    
    Args:
        path: Path to the JSON file to load
        sample_size: Number of items to return from the dataset.
                    If None, returns the full dataset.
                    Defaults to 10 for testing purposes.
    
    Returns:
        List of dictionaries containing the loaded JSON data
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        json.JSONDecodeError: If the file contains invalid JSON
        ValueError: If the sample size is larger than the dataset
    """
    try:
        # Convert string path to Path object for better handling
        file_path = Path(path)
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"No file found at {path}")
            
        # Open and read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
        # Validate that we got a list
        if not isinstance(json_data, list):
            raise ValueError(f"Expected a JSON array, but got {type(json_data)}")
            
        # Handle sampling if requested
        if sample_size is not None:
            if sample_size > len(json_data):
                raise ValueError(
                    f"Requested sample size {sample_size} is larger than "
                    f"dataset size {len(json_data)}"
                )
            return json_data[:sample_size]
            
        return json_data
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse JSON from {path}: {str(e)}", 
            e.doc, 
            e.pos
        )
    except Exception as e:
        raise Exception(f"Error loading JSON data from {path}: {str(e)}")

def table_to_string(table: List[List[str]]) -> str:
    """
    Convert a table (list of lists) to a string representation.
    This creates a formatted string where cells are separated by
    pipes (|) and rows are separated by newlines.
    
    Args:
        table: Table data as a list of lists, where each inner list
               represents a row
               
    Returns:
        String representation of the table
    """
    return "\n".join(" | ".join(str(cell) for cell in row) for row in table)