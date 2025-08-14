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
    Validates that BOTH train and test have:
    1. Same number of columns as schema.yaml
    2. Same column names as schema.yaml
    3. All required numerical & categorical columns present
    """
    try:
        schema = read_yaml_file("schema.yaml")

        # Get all columns from schema (support list or dict)
        schema_columns = schema.get("columns", [])
        if isinstance(schema_columns, dict):
            schema_columns = list(schema_columns.keys())
        elif isinstance(schema_columns, list):
            # Handle list of dicts: [{col: type}, {col: type}]
            if all(isinstance(c, dict) for c in schema_columns):
                schema_columns = [list(c.keys())[0] for c in schema_columns]
        if not schema_columns:
            logging.error("No 'columns' found in schema.yaml.")
            return False

        numerical_columns   = schema.get("numerical_columns", [])
        categorical_columns = schema.get("categorical_columns", [])

        if not numerical_columns and not categorical_columns:
            logging.error("No 'numerical_columns' or 'categorical_columns' found in schema.yaml.")
            return False

        status = True
        for df_name, df in zip(["train", "test"], [self.train_df, self.test_df]):
            df_cols = list(df.columns)

            # 1. Check number of columns
            if len(df_cols) != len(schema_columns):
                logging.error(f"[{df_name}] Column count mismatch: expected {len(schema_columns)}, got {len(df_cols)}")
                status = False

            # 2. Check exact column names match
            if set(df_cols) != set(schema_columns):
                missing_in_df = list(set(schema_columns) - set(df_cols))
                extra_in_df   = list(set(df_cols) - set(schema_columns))
                if missing_in_df:
                    logging.warning(f"[{df_name}] Missing columns: {missing_in_df}")
                if extra_in_df:
                    logging.warning(f"[{df_name}] Extra columns: {extra_in_df}")
                status = False

            # 3. Check numerical & categorical columns separately
            missing_num = [c for c in numerical_columns if c not in df_cols]
            missing_cat = [c for c in categorical_columns if c not in df_cols]

            if missing_num:
                logging.warning(f"[{df_name}] Missing numerical columns: {missing_num}")
                status = False
            if missing_cat:
                logging.warning(f"[{df_name}] Missing categorical columns: {missing_cat}")
                status = False

        if status:
            logging.info("✔ Both train and test match schema column count, names, and required types.")
        else:
            logging.error("✘ Train or test failed schema column validation.")
        return status

    except Exception as e:
        logging.error(f"Exception during validation: {e}")
        raise MyException(e, sys)
