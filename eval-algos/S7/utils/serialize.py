import json
import pandas as pd
from typing import Dict, Any


def serialize_analysis(analysis: Dict[str, Any], output_path: str) -> None:
    """
    Serializes the analysis dictionary to a JSON file.
    Only handles pandas DataFrames and regular dictionaries.

    Args:
        analysis (Dict[str, Any]): Analysis dictionary to serialize
        output_path (str): Path to save the JSON file
    """
    serializable_analysis = {}
    
    # Handle DataFrames
    for key, value in analysis.items():
        if isinstance(value, pd.DataFrame):
            serializable_analysis[key] = {
                "type": "DataFrame",
                "data": value.to_json(orient='split', date_format='iso')
            }
        else:
            # Skip dataclasses, only include regular data types
            try:
                json.dumps(value)  # Test if value is JSON serializable
                serializable_analysis[key] = value
            except (TypeError, OverflowError):
                continue
    
    with open(output_path, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)


def deserialize_analysis(input_path: str) -> Dict[str, Any]:
    """
    Deserializes an analysis dictionary from a JSON file.
    Only reconstructs pandas DataFrames.

    Args:
        input_path (str): Path to the JSON file

    Returns:
        Dict[str, Any]: Reconstructed analysis dictionary
    """
    with open(input_path, 'r') as f:
        serialized = json.load(f)
    
    analysis = {}
    for key, value in serialized.items():
        if isinstance(value, dict) and "type" in value:
            if value["type"] == "DataFrame":
                analysis[key] = pd.read_json(value["data"], orient='split')
        else:
            analysis[key] = value
    
    return analysis 