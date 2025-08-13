from logger import logging
from exception import MyException
from components.data_ingestion import DataIngestion



class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()




    def start_data_ingestion(self):
        try:
            logging.info("Starting data ingestion process")
            self.data_ingestion.run()
        except MyException as e:
            logging.error(f"Error occurred while starting data ingestion: {e}")