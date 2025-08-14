
import os
import sys
import pandas as pd
import joblib
from logger import logging
from exception import MyException
from components.data_transformation import DataTransformation
from utils.main_utils import read_yaml_file

class PredictionPipeline:
	def __init__(self):
		logging.info("PredictionPipeline initialized")

	def predict_from_df(self, input_df: pd.DataFrame):
		"""
		Accepts input data as a pandas DataFrame, applies transformations, and returns predictions.
		If preprocessor.pkl is not available, fits the preprocessor on training data.
		"""
		try:
			# Find latest timestamp
			base_dir = "artifacts"
			timestamps = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
			if not timestamps:
				raise FileNotFoundError("No timestamped artifacts found.")
			latest_timestamp = sorted(timestamps)[-1]
			model_dir = os.path.join(base_dir, latest_timestamp, "model_trainer")
			transformation_dir = os.path.join(base_dir, latest_timestamp, "data_transformation")

			# Load model
			model_path = os.path.join(model_dir, "random_forest_model.pkl")
			if not os.path.exists(model_path):
				raise FileNotFoundError(f"Model not found at {model_path}")
			model = joblib.load(model_path)

			# Try to load fitted preprocessor
			preprocessor_path = os.path.join(transformation_dir, "preprocessor.pkl")
			dt = DataTransformation()
			dt.prepare_data_transformation()  # loads schema and train/test
			if os.path.exists(preprocessor_path):
				preprocessor = joblib.load(preprocessor_path)
			else:
				# Fit preprocessor on training data if pickle not available
				preprocessor = dt.get_data_transformer_object()
				input_feature_train_df = dt.train_df.drop(columns=[read_yaml_file("schema.yaml").get("target_column")], axis=1)
				for func in [dt._map_gender_column, dt._drop_id_column, dt._create_dummy_columns, dt._rename_columns]:
					input_feature_train_df = func(input_feature_train_df)
				preprocessor.fit(input_feature_train_df)

			# Apply custom transformations to input
			for func in [dt._map_gender_column, dt._drop_id_column, dt._create_dummy_columns, dt._rename_columns]:
				input_df = func(input_df)
			logging.info("Custom transformations applied to input data")

			# Get expected columns from training transformation
			input_feature_train_df = dt.train_df.drop(columns=[read_yaml_file("schema.yaml").get("target_column")], axis=1)
			for func in [dt._map_gender_column, dt._drop_id_column, dt._create_dummy_columns, dt._rename_columns]:
				input_feature_train_df = func(input_feature_train_df)
			expected_columns = input_feature_train_df.columns.tolist()

			# Add missing columns with default value 0
			missing_cols = set(expected_columns) - set(input_df.columns)
			if missing_cols:
				logging.warning(f"Missing columns in input: {missing_cols}. Filling with default 0 values.")
				for col in missing_cols:
					input_df[col] = 0

			# Remove extra columns
			extra_cols = set(input_df.columns) - set(expected_columns)
			if extra_cols:
				logging.info(f"Extra columns in input: {extra_cols}. Dropping them.")
				input_df = input_df.drop(columns=list(extra_cols))

			# Reorder columns to match expected order
			input_df = input_df[expected_columns]

			# Transform input data
			input_arr = preprocessor.transform(input_df)
			logging.info("Input data transformed")

			# Predict
			predictions = model.predict(input_arr)
			logging.info("Predictions generated")
			return predictions.tolist()
		except Exception as e:
			logging.error(f"Error in prediction pipeline: {e}")
			raise MyException(e, sys)
