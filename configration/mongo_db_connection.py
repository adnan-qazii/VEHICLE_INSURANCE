import os
import sys
import pymongo
from dotenv import load_dotenv
from logger import logging
from exception import MyException
import certifi

load_dotenv()


class MongoDBConnection:
    def __init__(self):
        try:
            logging.info("Starting MongoDB connection...")
            self.client = pymongo.MongoClient(
                os.getenv("CONNECTION_URL"),
                tls=True,
                tlsCAFile=os.getenv("TLS_CA_FILE")  # path from env
            )
            self.db = self.client[os.getenv("DB_NAME")]
            logging.info(
                "MongoDB connection established successfully with database: %s",
                os.getenv("DB_NAME")
            )
        except Exception as e:
            logging.error(f"Error occurred while connecting to MongoDB: {e}")
            raise MyException("MongoDB connection failed", sys) from e
