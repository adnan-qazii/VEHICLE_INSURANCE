from logger import logging
import os
import sys
from data_acess.data_acess import DataAccess
import pandas as pd
from sklearn.model_selection import train_test_split
from exception import MyException
from dotenv import load_dotenv
from constants import train_test_split_ratio

load_dotenv()


import datetime

class DataIngestion:
	def __init__(self):
		self.collection_name = os.getenv("COLLECTION_NAME")
		self.train_test_split_ratio = train_test_split_ratio
		self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		self.base_dir = os.path.join("artifacts", self.timestamp, "dataingestion")

	def fetch_and_save_raw_data(self) -> pd.DataFrame:
		try:
			logging.info(f"Fetching data from MongoDB collection: {self.collection_name}")
			data_access = DataAccess()
			df = data_access.fetch_data(collection_name=self.collection_name)
			logging.info(f"Fetched data shape: {df.shape}")
			raw_dir = os.path.join(self.base_dir, "raw")
			os.makedirs(raw_dir, exist_ok=True)
			raw_file_path = os.path.join(raw_dir, "raw_data.csv")
			df.to_csv(raw_file_path, index=False, header=True)
			logging.info(f"Raw data saved to {raw_file_path}")
			return df
		except Exception as e:
			logging.error(f"Error in fetch_and_save_raw_data: {e}")
			raise MyException(e, sys)

	def split_and_save_train_test(self, df: pd.DataFrame):
		try:
			train_set, test_set = train_test_split(df, test_size=self.train_test_split_ratio)
			split_dir = os.path.join(self.base_dir, "split")
			train_dir = os.path.join(split_dir, "train")
			test_dir = os.path.join(split_dir, "test")
			os.makedirs(train_dir, exist_ok=True)
			os.makedirs(test_dir, exist_ok=True)
			train_file_path = os.path.join(train_dir, "train.csv")
			test_file_path = os.path.join(test_dir, "test.csv")
			train_set.to_csv(train_file_path, index=False, header=True)
			test_set.to_csv(test_file_path, index=False, header=True)
			logging.info(f"Train data saved to {train_file_path}")
			logging.info(f"Test data saved to {test_file_path}")
		except Exception as e:
			logging.error(f"Error in split_and_save_train_test: {e}")
			raise MyException(e, sys)

	def run(self):
		df = self.fetch_and_save_raw_data()
		self.split_and_save_train_test(df)
    
