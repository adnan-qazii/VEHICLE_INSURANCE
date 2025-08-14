import os
import sys
import numpy as np
import pandas as pd
from logger import logging
from exception import MyException
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from utils.main_utils import read_yaml_file


class DataTransformation:
    def __init__(self):
        """
        Loads train and test DataFrames from the latest timestamped artifacts directory.
        Also loads the schema configuration once and stores it on the instance.
        """
        try:
            # Load schema once and keep it
            schema_path = "schema.yaml"
            logging.info(f"Loading schema from: {schema_path}")
            self._schema_config = read_yaml_file(schema_path) or {}
            logging.info(f"Schema keys: {list(self._schema_config.keys())}")

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

            self.train_df = pd.read_csv(train_path)
            self.test_df = pd.read_csv(test_path)

            logging.info(f"Train data shape: {self.train_df.shape}")
            logging.info(f"Test data shape: {self.test_df.shape}")

        except Exception as e:
            logging.error(f"Error in DataTransformation initialization: {e}")
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer Pipeline:
        - StandardScaler on numeric features (num_features)
        - MinMaxScaler on specified columns (mm_columns)
        - passthrough for remaining columns
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            # Initialize transformers
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()
            logging.info("Transformers initialized: StandardScaler & MinMaxScaler")

            # Load schema-configured column groups
            num_features = self._schema_config.get("num_features", [])
            mm_columns = self._schema_config.get("mm_columns", [])
            logging.info(f"num_features from schema: {num_features}")
            logging.info(f"mm_columns from schema: {mm_columns}")

            # Optional: prevent overlap between the two groups (avoids double scaling)
            mm_columns = [c for c in mm_columns if c not in num_features]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("standard_scaler", numeric_transformer, num_features),
                    ("minmax_scaler", min_max_scaler, mm_columns),
                ],
                remainder="passthrough"
            )

            final_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
            logging.info("Final Pipeline ready.")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object")
            raise MyException(e, sys) from e

    def _map_gender_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map Gender column to 0 for Female and 1 for Male.
        Leaves values as-is if column doesn't exist.
        """
        logging.info("Mapping 'Gender' column to binary values")
        if "Gender" in df.columns:
            df = df.copy()
            df["Gender"] = (
                df["Gender"].map({"Female": 0, "Male": 1})
                .fillna(df["Gender"])  # in case of unexpected categories/NaNs
            )
            # If mapping succeeds for all rows, cast to int
            if pd.api.types.is_numeric_dtype(df["Gender"]):
                # If any non-integer numeric values exist due to NaNs, coerce safely
                if df["Gender"].isna().any():
                    df["Gender"] = df["Gender"].fillna(-1)
                df["Gender"] = df["Gender"].astype(int)
        else:
            logging.info("'Gender' column not found; skipping mapping.")
        return df

    def _create_dummy_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dummy variables for categorical features (drop_first=True)."""
        logging.info("Creating dummy variables for categorical features")
        return pd.get_dummies(df, drop_first=True)

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename specific columns and ensure integer types for selected dummy columns."""
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype(int)
        return df

    def _drop_id_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the column specified by 'drop_columns' in schema (if present)."""
        logging.info("Dropping ID/identifier column if configured in schema")
        drop_col = self._schema_config.get("drop_columns")
        if drop_col is None:
            logging.info("No 'drop_columns' key in schema; skipping drop.")
            return df
        # Allow either a single column name or a list in the schema
        if isinstance(drop_col, str):
            drop_cols = [drop_col]
        else:
            drop_cols = list(drop_col)

        existing = [c for c in drop_cols if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
            logging.info(f"Dropped columns: {existing}")
        else:
            logging.info("No configured drop columns present in DataFrame.")
        return df



    def initiate_data_transformation(self, target_column: str = "target"):
        """
        Transforms train and test data, saves numpy arrays in data_transformation folder under latest timestamp.
        """
        try:
            logging.info("Data Transformation Started !!!")

            # Split input/target features
            input_feature_train_df = self.train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = self.train_df[target_column]
            input_feature_test_df = self.test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = self.test_df[target_column]
            logging.info("Input and Target cols defined for both train and test df.")

            # Apply custom transformations
            for func in [self._map_gender_column, self._drop_id_column, self._create_dummy_columns, self._rename_columns]:
                input_feature_train_df = func(input_feature_train_df)
                input_feature_test_df = func(input_feature_test_df)
            logging.info("Custom transformations applied to train and test data")

            # Data transformation
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")

            # Optionally apply SMOTE or other balancing here if needed
            # Concatenate features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("feature-target concatenation done for train-test df.")

            # Save numpy arrays in data_transformation folder under latest timestamp
            transformation_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.train_df.attrs.get('filepath_or_buffer', '')))), "data_transformation")
            # Fallback to artifact_dir if above fails
            if not os.path.exists(transformation_dir):
                transformation_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "artifacts", "data_transformation")
            os.makedirs(transformation_dir, exist_ok=True)
            train_np_path = os.path.join(transformation_dir, "train.npy")
            test_np_path = os.path.join(transformation_dir, "test.npy")
            np.save(train_np_path, train_arr)
            np.save(test_np_path, test_arr)
            logging.info(f"Saved train numpy array at: {train_np_path}")
            logging.info(f"Saved test numpy array at: {test_np_path}")

            logging.info("Data transformation completed successfully")
            return train_np_path, test_np_path
        except Exception as e:
            logging.error(f"Error in initiate_data_transformation: {e}")
            raise MyException(e, sys)


        