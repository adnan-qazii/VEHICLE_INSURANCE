#test logging and exception handling


'''
from logger import logging
from exception import MyException
import sys

num1=5
num2=0
logging.info("Starting division operation")
try:
    result = num1 + num2
    logging.info(f"Division result: {result}")
except Exception as e:
    raise MyException("Error occurred", sys) from e

    '''


# checking data base mongo db connection
'''
import os
from dotenv import load_dotenv
from logger import logging
from configration.mongo_db_connection import MongoDBConnection


mongo_db_connection = MongoDBConnection()


# Choose the collection name you want to query
collection_name = os.getenv("COLLECTION_NAME")

# Fetch first 10 documents
documents = mongo_db_connection.db[collection_name].find().limit(10)

# Print them
for doc in documents:
    logging.info(doc)
    '''

#checking data acess 
'''
from logger import logging
from data_acess.data_acess import DataAccess
logging.info("Starting data access operation")
data = DataAccess()
logging.info("Data access operation completed")


data=DataAccess()
df=data.fetch_data("Proj1_Data")

print(df.shape)
print(df.head())
'''



#test data ingestion
'''
from components.data_ingestion import DataIngestion


exp= DataIngestion()
exp.run()
'''



# testing training pipeline
'''
from logger import logging
from pipeline.training_pipeline import TrainingPipeline
logging.info("Starting training pipeline")
start= TrainingPipeline()
start.start_data_ingestion()
logging.info("Training pipeline completed")
'''