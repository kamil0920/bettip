"""Data parsing utilities for preprocessing."""
import json
import pandas as pd
from typing import Any, Dict, Optional, List, Union

from src.preprocessing.interfaces import IDataParser

class JSONParser(IDataParser):
    """Parser for JSON data."""
    def parse(self, raw_data: Any) -> Optional[Dict[str, Any]]:
        """Parses JSON data - supports strings and dicts."""
        if raw_data is None:
            return None

        if isinstance(raw_data, dict):
            return raw_data

        if isinstance(raw_data, str):
            s = raw_data.strip()
            if not s:
                return None

            if s[0] not in ('{', '[') or s[-1] not in ('}', ']'):
                return None

            try:
                return json.loads(raw_data)
            except json.JSONDecodeError:
                try:
                    unescaped = raw_data.encode('utf-8').decode('unicode_escape')
                    return json.loads(unescaped)
                except Exception:
                    return None
        return None


class ParquetParser(IDataParser):
    """
    Parser for Parquet data.
    Handles inputs that are already Dicts (from pandas.to_dict) or pandas Series/DataFrames.
    """

    def parse(self, raw_data: Any) -> Optional[Dict[str, Any]]:
        """
        Parses Parquet data into a standard dictionary format.

        Accepts:
        - dict (already parsed)
        - pd.Series (single row)
        - pd.DataFrame (single row only, otherwise fails or returns first)
        """
        if raw_data is None:
            return None

        if isinstance(raw_data, dict):
            return raw_data

        if isinstance(raw_data, pd.Series):
            return raw_data.to_dict()

        if isinstance(raw_data, pd.DataFrame):
            if raw_data.empty:
                return None
            if len(raw_data) == 1:
                return raw_data.iloc[0].to_dict()

            return raw_data.iloc[0].to_dict()

        return None
