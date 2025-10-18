import json
from typing import Any, Dict, Optional

from interfaces import IDataParser


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
            if not s or s[0] not in ('{', '[') or s[-1] not in ('}', ']'):
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