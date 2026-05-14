import logging
import os
from typing import Optional

import pandas as pd

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataIngestor:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Loads the air compressor CSV file from the data/raw directory."""
        logger.info(f"Attempting to load data from: {self.raw_data_path}")

        if not os.path.exists(self.raw_data_path):
            logger.error(f"File not found: {self.raw_data_path}")
            raise FileNotFoundError(f"Check your data path: {self.raw_data_path}")

        try:
            self.df = pd.read_csv(self.raw_data_path)
            logger.info(f"Successfully loaded dataframe with {self.df.shape[0]} rows.")
            return self.df
        except Exception as e:
            logger.error(f"Error occurred while reading CSV: {e}")
            raise

    def get_data_summary(self):
        """Prints a basic summary of the ingested data for verification."""
        if self.df is not None:
            logger.info("Data Summary:")
            logger.info(f"Columns: {list(self.df.columns)}")
            logger.info(f"Missing values: \n{self.df.isnull().sum()}")
        else:
            logger.warning("No data loaded yet. Call load_data() first.")


if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "raw", "aircompressor.csv")

    ingestor = DataIngestor(DATA_PATH)
    data = ingestor.load_data()
    ingestor.get_data_summary()
