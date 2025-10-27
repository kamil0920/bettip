from typing import Dict

import pandas as pd
from interfaces import IDataLoader


class ParquetDataLoader(IDataLoader):

    def load(self, filepath: str) -> pd.DataFrame:
        """
        Load data from parquet file

        Args:
            filepath: Path to parquet file

        Returns:
            DataFrame with data
        """
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Loaded {filepath}: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"✗ Error {filepath}: {str(e)}")
            raise


class MultiFileLoader:

    def __init__(self, loader: IDataLoader):
        self.loader = loader

    def load_all(self, filepaths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Load all files in filepaths

        Args:
            filepaths: dict {name: path}

        Returns:
            Dict {name: DataFrame}
        """
        data = {}
        for name, path in filepaths.items():
            data[name] = self.loader.load(path)
        return data
