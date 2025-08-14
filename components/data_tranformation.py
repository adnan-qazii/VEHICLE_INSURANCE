import os
import sys
import numpy as np
import pandas as pd
from components.data_validation import DataValidation
from exception import MyException
from imblearn.combine import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer 
from utils.main_utils import read_yaml_file


class DataTransformation:
    def __init__(self):
        """
        Loads train and test DataFrames from the latest timestamped artifacts directory.
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
            artifact_dir = os.path.join(base_dir, latest_timestamp)
            split_dir = os.path.join(artifact_dir, "dataingestion", "split")
            train_path = os.path.join(split_dir, "train", "train.csv")
            test_path = os.path.join(split_dir, "test", "test.csv")
            logging.info(f"Loading train data from: {train_path}")
            logging.info(f"Loading test data from: {test_path}")
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                logging.error("Train or test CSV file not found in latest split directory.")
                raise FileNotFoundError("Train or test CSV file not found in latest split directory.")
            import pandas as pd
            self.train_df = pd.read_csv(train_path)
            self.test_df = pd.read_csv(test_path)
            logging.info(f"Train data shape: {self.train_df.shape}")
            logging.info(f"Test data shape: {self.test_df.shape}")
        except Exception as e:
            logging.error(f"Error in DataTransformation initialization: {e}")
            raise MyException(e, sys)
        

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize transformers
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()
            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")

            # Load schema configurations
            num_features = read_yaml_file("schema.yaml").get("num_features", [])
            mm_columns = read_yaml_file("schema.yaml").get("mm_columns", [])
            logging.info("Cols loaded from schema.")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, num_features),
                    ("MinMaxScaler", min_max_scaler, mm_columns)
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e



    def _map_gender_column(self, df):
        """Map Gender column to 0 for Female and 1 for Male."""
        logging.info("Mapping 'Gender' column to binary values")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df

    def _create_dummy_columns(self, df):
        """Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first=True)
        return df

    def _rename_columns(self, df):
        """Rename specific columns and ensure integer types for dummy columns."""
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df

    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        drop_col = self._schema_config['drop_columns']
        if drop_col in df.columns:
            df = df.drop(drop_col, axis=1)
        return df
