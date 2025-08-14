import os
import sys
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, classification_report
from logger import logging
from exception import MyException

class ModelEvaluation:
	def __init__(self):
		logging.info("ModelEvaluation initialized")

	def evaluate_model(self):
		"""
		Loads model and test numpy array from latest timestamped directory, evaluates, and saves report in model_evaluation directory.
		"""
		try:
			base_dir = "artifacts"
			timestamps = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
			latest_timestamp = sorted(timestamps)[-1]
			transformation_dir = os.path.join(base_dir, latest_timestamp, "data_transformation")
			test_np_path = os.path.join(transformation_dir, "test.npy")
			model_dir = os.path.join(base_dir, latest_timestamp, "model_trainer")
			model_path = os.path.join(model_dir, "random_forest_model.pkl")

			logging.info(f"Loading test numpy array from: {test_np_path}")
			test_arr = np.load(test_np_path)
			X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

			logging.info(f"Loading model from: {model_path}")
			model = joblib.load(model_path)

			logging.info("Evaluating model...")
			y_pred = model.predict(X_test)
			acc = accuracy_score(y_test, y_pred)
			prec = precision_score(y_test, y_pred, average='weighted')
			report = classification_report(y_test, y_pred)
			logging.info(f"Accuracy: {acc}")
			logging.info(f"Precision: {prec}")
			logging.info(f"Classification Report:\n{report}")

			# Save evaluation report
			eval_dir = os.path.join(base_dir, latest_timestamp, "model_evaluation")
			os.makedirs(eval_dir, exist_ok=True)
			import yaml
			report_path = os.path.join(eval_dir, "model_evaluation_report.yaml")
			with open(report_path, "w") as f:
				yaml.dump({
					"accuracy": float(acc),
					"precision": float(prec),
					"classification_report": report,
					"model_path": model_path,
					"test_numpy_path": test_np_path
				}, f)
			logging.info(f"Evaluation report saved at: {report_path}")

			return report_path
		except Exception as e:
			logging.error(f"Error in model evaluation: {e}")
			raise MyException(e, sys)



	def run(self) -> None:
		self.evaluate_model()
		