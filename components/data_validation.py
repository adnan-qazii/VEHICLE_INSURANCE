import os
import sys
import pandas as pd
from logger import logging
from exception import MyException
from utils.main_utils import read_yaml_file



class DataValidation:
    
    """
    Handles loading and validating train/test datasets against schema.
    """
    def __init__(self):
        """
        Loads the latest train and test datasets from artifacts directory.
        """
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

    def validate_number_of_columns(self) -> bool:
        """
        Validates that both train and test DataFrames contain all numerical and categorical columns as per schema.yaml.
        Logs missing columns for each DataFrame and returns True only if all required columns are present.
        """
        try:
            schema = read_yaml_file("schema.yaml")
            numerical_columns = schema.get("numerical_columns", [])
            categorical_columns = schema.get("categorical_columns", [])
            if not numerical_columns and not categorical_columns:
                logging.error("No numerical or categorical columns found in schema.yaml.")
                return False

            status = True
            for df_name, df in zip(["train", "test"], [self.train_df, self.test_df]):
                missing_numerical = [col for col in numerical_columns if col not in df.columns]
                missing_categorical = [col for col in categorical_columns if col not in df.columns]

                if missing_numerical:
                    logging.warning(f"{df_name.capitalize()} data missing numerical columns: {missing_numerical}")
                    status = False
                if missing_categorical:
                    logging.warning(f"{df_name.capitalize()} data missing categorical columns: {missing_categorical}")
                    status = False

            if status:
                logging.info("Both train and test data contain all required numerical and categorical columns.")
            else:
                logging.error("Train or test data missing required columns as per schema.yaml.")
            return status
        except Exception as e:
            logging.error(f"Exception during column validation: {e}")
            raise MyException(e, sys)
        
    def check_numerical_and_categorical_columns(self) -> bool:
        """
        Checks if train and test data have all numerical and categorical columns as per schema.yaml.
        Returns True if all columns exist, else logs missing columns and returns False.
        """
        try:
            schema = read_yaml_file("schema.yaml")
            numerical_columns = schema.get("numerical_columns", [])
            categorical_columns = schema.get("categorical_columns", [])
            status = True

            for df_name, df in zip(["train", "test"], [self.train_df, self.test_df]):
                dataframe_columns = df.columns
                missing_numerical_columns = [col for col in numerical_columns if col not in dataframe_columns]
                missing_categorical_columns = [col for col in categorical_columns if col not in dataframe_columns]

                if missing_numerical_columns:
                    logging.info(f"Missing numerical columns in {df_name} data: {missing_numerical_columns}")
                    status = False
                if missing_categorical_columns:
                    logging.info(f"Missing categorical columns in {df_name} data: {missing_categorical_columns}")
                    status = False

            return status
        except Exception as e:
            logging.error(f"Error during numerical/categorical column check: {e}")
            raise MyException(e, sys)
        

    
    

    

    def run(self):
        # run all fxn in data validation class

        validation= DataValidation()
        validation.validate_number_of_columns()
        validation.check_numerical_and_categorical_columns()

    