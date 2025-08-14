from logger import logging
from exception import MyException
from components.data_ingestion import DataIngestion
from components.data_validation import DataValidation
from components.data_transformation import DataTransformation



class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()




    def start_data_ingestion(self):
        try:
            logging.info("Starting data ingestion process")
            self.data_ingestion.run()
        except MyException as e:
            logging.error(f"Error occurred while starting data ingestion: {e}")





    def start_data_validation(self):
        try:
            logging.info("Starting data validation process")
            self.data_validation.run()
        except MyException as e:
            logging.error(f"Error occurred while starting data validation: {e}")


    def start_data_transformation(self):
        try:
            logging.info("Starting data transformation process")
            self.data_transformation.run()
        except MyException as e:
            logging.error(f"Error occurred while starting data transformation: {e}")



            