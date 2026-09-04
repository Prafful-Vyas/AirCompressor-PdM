import logging
import os

import pandas as pd

from .config import DROP_COLS, RAW_FEATURE_COLS

# All columns the raw CSV must have: sensor readings, plus everything
# train.py drops before fitting (id, acmotor, and the fault targets).
REQUIRED_COLS = sorted(set(RAW_FEATURE_COLS) | set(DROP_COLS))

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataIngestor:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.df: pd.DataFrame | None = None

    def load_data(self) -> pd.DataFrame:
        """Loads the air compressor CSV file from the data/raw directory."""
        logger.info(f"Attempting to load data from: {self.raw_data_path}")

        if not os.path.exists(self.raw_data_path):
            logger.error(f"File not found: {self.raw_data_path}")
            raise FileNotFoundError(f"Check your data path: {self.raw_data_path}")

        try:
            df = pd.read_csv(self.raw_data_path)
        except Exception as e:
            logger.error(f"Error occurred while reading CSV: {e}")
            raise

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        non_numeric = [
            c
            for c in RAW_FEATURE_COLS
            if pd.to_numeric(df[c], errors="coerce").isna().sum() > df[c].isna().sum()
        ]
        if non_numeric:
            raise ValueError(
                f"Non-numeric values found in sensor columns: {non_numeric}"
            )

        self.df = df
        logger.info(f"Successfully loaded dataframe with {self.df.shape[0]} rows.")
        return self.df

    def get_data_summary(self):
        """Prints a basic summary of the ingested data for verification."""
        if self.df is not None:
            logger.info("Data Summary:")
            logger.info(f"Columns: {list(self.df.columns)}")
            logger.info(f"Missing values: \n{self.df.isnull().sum()}")
        else:
            logger.warning("No data loaded yet. Call load_data() first.")


if __name__ == "__main__":
    from src.settings import get_settings

    ingestor = DataIngestor(get_settings().data_raw_path)
    data = ingestor.load_data()
    ingestor.get_data_summary()
