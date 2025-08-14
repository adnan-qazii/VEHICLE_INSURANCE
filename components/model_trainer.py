
import sys
from sklearn.ensemble import RandomForestClassifier
from logger import logging
from exception import MyException
import numpy as np
import joblib
import os
from constants import *



class ModelTrainer():
    def __init__(self):
        logging.info("ModelTrainer initialized")



    def initiate_model_training(self):
        """
        Train RandomForest model using train numpy array from latest timestamped directory, save model in artifacts/model_trainer.
        """
        try:
            logging.info("Model training started...")
            # Get train numpy path from latest timestamped directory
            base_dir = "artifacts"
            timestamps = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            latest_timestamp = sorted(timestamps)[-1]
            transformation_dir = os.path.join(base_dir, latest_timestamp, "data_transformation")
            train_np_path = os.path.join(transformation_dir, "train.npy")
            logging.info(f"Loading train numpy array from: {train_np_path}")
            train_arr = np.load(train_np_path)
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]

            logging.info("Training RandomForestClassifier...")
            model = RandomForestClassifier(
                n_estimators=MODEL_TRAINER_N_ESTIMATORS,
                min_samples_split=MODEL_TRAINER_MIN_SAMPLES_SPLIT,
                min_samples_leaf=MODEL_TRAINER_MIN_SAMPLES_LEAF,
                max_depth=MIN_SAMPLES_SPLIT_MAX_DEPTH,
                criterion=MIN_SAMPLES_SPLIT_CRITERION,
                random_state=MIN_SAMPLES_SPLIT_RANDOM_STATE
            )
            model.fit(X_train, y_train)

            # Save model in artifacts/model_trainer
            model_dir = os.path.join(base_dir, latest_timestamp, "model_trainer")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "random_forest_model.pkl")
            joblib.dump(model, model_path)
            logging.info(f"Model saved at: {model_path}")

            return model_path
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise MyException(e, sys)
        


    def run(self):
        self.initiate_model_training()


run= ModelTrainer()
run.run()