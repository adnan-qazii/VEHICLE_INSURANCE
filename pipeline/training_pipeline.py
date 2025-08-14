from logger import logging
from exception import MyException
from components.data_ingestion import DataIngestion
from components.data_validation import DataValidation
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer



class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()




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



    def start_model_training(self):
        try:
            logging.info("Starting model training process")
            self.model_trainer.run()
        except MyException as e:
            logging.error(f"Error occurred while starting model training: {e}")


    def run(self) -> None:
        """Run the full training pipeline in order."""
        logging.info("Training pipeline started")
        
        self.start_model_training()
        logging.info("Training pipeline finished successfully")

            