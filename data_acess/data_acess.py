import sys
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from configration.mongo_db_connection import MongoDBConnection
from logger import logging
from exception import MyException




class DataAccess:

    def __init__(self):
        load_dotenv()
        self.mongo_db_connection = MongoDBConnection()

    def fetch_data(self, collection_name, query=None):
        try:
            if query is None:
                query = {}
            documents = self.mongo_db_connection.db[collection_name].find(query, {"_id": 0})
            return pd.DataFrame(list(documents))
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise MyException("Error fetching data", sys) from e
