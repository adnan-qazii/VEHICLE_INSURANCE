import os
import sys
import pandas as pd
import joblib
from logger import logging
from exception import MyException
from components.data_transformation import DataTransformation

class PredictionPipeline:
	def __init__(self):
		logging.info("PredictionPipeline initialized")

	def predict_from_df(self, input_df: pd.DataFrame):
		"""
		Accepts input data as a pandas DataFrame, applies transformations, and returns predictions.
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

			# Load preprocessor from DataTransformation
			dt = DataTransformation()
			dt.prepare_data_transformation()  # loads schema
			preprocessor = dt.get_data_transformer_object()

			# Apply custom transformations
			for func in [dt._map_gender_column, dt._drop_id_column, dt._create_dummy_columns, dt._rename_columns]:
				input_df = func(input_df)
			logging.info("Custom transformations applied to input data")

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
