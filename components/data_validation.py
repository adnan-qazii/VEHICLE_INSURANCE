import os
import sys
import pandas as pd
from logger import logging
from exception import MyException
 
class DataValidation:
    def __init__(self):
        try:
            base_dir = "artifacts"
            logging.info(f"Looking for artifacts directory at: {base_dir}")
            if not os.path.exists(base_dir):
                logging.error("Artifacts directory not found.")
                raise FileNotFoundError("Artifacts directory not found.")
            timestamps = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if not timestamps:
                logging.error("No timestamped directories found in artifacts.")
                raise FileNotFoundError("No timestamped directories found in artifacts.")
            latest_timestamp = sorted(timestamps)[-1]
            split_dir = os.path.join(base_dir, latest_timestamp, "dataingestion", "split")
            train_path = os.path.join(split_dir, "train", "train.csv")
            test_path = os.path.join(split_dir, "test", "test.csv")
            logging.info(f"Loading train data from: {train_path}")
            logging.info(f"Loading test data from: {test_path}")
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                logging.error("Train or test CSV file not found in latest split directory.")
                raise FileNotFoundError("Train or test CSV file not found in latest split directory.")
            self.train_df = pd.read_csv(train_path)
            self.test_df = pd.read_csv(test_path)
            logging.info(f"Train data shape: {self.train_df.shape}")
            logging.info(f"Test data shape: {self.test_df.shape}")
        except Exception as e:
            logging.error(f"Error in DataValidation initialization: {e}")
            raise MyException(e, sys)


    def validate_number_of_columns(self,dataframe :pd.DataFrame)-> bool:
        try:
            logging.info("Validating train and test data.")
            if self.train_df.empty or self.test_df.empty:
                logging.error("Train or test DataFrame is empty.")
                raise ValueError("Train or test DataFrame is empty.")
            # Add more validation checks as needed
            logging.info("Data validation completed successfully.")
        except Exception as e:
            logging.error(f"Error during data validation: {e}")
            raise MyException(e, sys)

