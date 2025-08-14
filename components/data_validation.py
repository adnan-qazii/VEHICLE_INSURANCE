import os
import sys
from pathlib import Path
import pandas as pd
import yaml

from logger import logging
from exception import MyException
from utils.main_utils import read_yaml_file


class DataValidation:
    """
    Loads the latest train/test datasets under artifacts/<timestamp>/dataingestion/split
    and validates them against a schema.yaml.
    """
    def __init__(self):
        "initialized "


    def prepare_data_validation(self, artifacts_dir: str = "artifacts", schema_path: str = "schema.yaml"):
        try:
            self.schema_path = schema_path
            base_dir = Path(artifacts_dir)
            logging.info(f"Looking for artifacts directory at: {base_dir.resolve()}")
            if not base_dir.exists():
                raise FileNotFoundError(f"Artifacts directory not found at {base_dir!s}")

            # choose most-recent directory by mtime
            subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
            if not subdirs:
                raise FileNotFoundError("No timestamped directories found in artifacts.")
            latest_dir = max(subdirs, key=lambda p: p.stat().st_mtime)
            self.artifact_dir = latest_dir

            split_dir = self.artifact_dir / "dataingestion" / "split"
            train_path = split_dir / "train" / "train.csv"
            test_path  = split_dir / "test" / "test.csv"

            logging.info(f"Loading train data from: {train_path}")
            logging.info(f"Loading test data from:  {test_path}")

            if not train_path.exists() or not test_path.exists():
                raise FileNotFoundError("Train or test CSV file not found in latest split directory.")

            self.train_df = pd.read_csv(train_path)
            self.test_df  = pd.read_csv(test_path)

            logging.info(f"Train shape: {self.train_df.shape} | Test shape: {self.test_df.shape}")
        except Exception as e:
            logging.error(f"Error in DataValidation initialization: {e}")
            raise MyException(e, sys)

    def _load_schema(self) -> dict:
        try:
            schema = read_yaml_file(self.schema_path)
            if not isinstance(schema, dict):
                raise ValueError("Schema file did not parse to a dict.")
            return schema
        except Exception as e:
            logging.error(f"Failed to read schema at {self.schema_path}: {e}")
            raise

    @staticmethod
    def _normalize_schema_columns(schema: dict):
        cols = schema.get("columns", [])
        if isinstance(cols, dict):
            schema_columns = list(cols.keys())
            types = cols
        elif isinstance(cols, list):
            if all(isinstance(c, dict) for c in cols):
                schema_columns = [list(c.keys())[0] for c in cols]
                types = {list(c.keys())[0]: list(c.values())[0] for c in cols}
            else:
                schema_columns = cols
                types = {}
        else:
            schema_columns, types = [], {}

        num_cols = schema.get("numerical_columns", []) or []
        cat_cols = schema.get("categorical_columns", []) or []
        return schema_columns, types, num_cols, cat_cols

    def validate_number_of_columns(self) -> dict:
        """
        Returns a detailed report dict, including pass/fail for train/test.
        """
        try:
            schema = self._load_schema()
            schema_columns, type_map, numerical_columns, categorical_columns = self._normalize_schema_columns(schema)

            if not schema_columns:
                msg = "No 'columns' found in schema.yaml."
                logging.error(msg)
                return {"ok": False, "error": msg}

            report = {"ok": True, "splits": {}}

            for name, df in (("train", self.train_df), ("test", self.test_df)):
                df_cols = list(df.columns)

                split_rep = {
                    "expected_count": len(schema_columns),
                    "actual_count": len(df_cols),
                    "missing": list(set(schema_columns) - set(df_cols)),
                    "extra": list(set(df_cols) - set(schema_columns)),
                    "missing_numerical": [c for c in numerical_columns if c not in df_cols],
                    "missing_categorical": [c for c in categorical_columns if c not in df_cols],
                    "order_matches": df_cols == schema_columns  # True if exact order match
                }

                # pass criteria: same set of columns AND counts match
                split_ok = (
                    split_rep["expected_count"] == split_rep["actual_count"]
                    and not split_rep["missing"]
                    and not split_rep["extra"]
                )

                # Optional: basic dtype checks if schema provided types
                dtype_issues = {}
                if type_map:
                    for col, expected in type_map.items():
                        if col in df.columns:
                            # very light check: pandas kind grouping
                            k = df[col].dtype.kind
                            if expected in ("int", "integer") and k not in ("i", "u"):
                                dtype_issues[col] = f"expected int, got {df[col].dtype}"
                                split_ok = False
                            elif expected in ("float", "double") and k not in ("f",):
                                dtype_issues[col] = f"expected float, got {df[col].dtype}"
                                split_ok = False
                            elif expected in ("string", "object", "category") and k not in ("O", "U", "S"):
                                dtype_issues[col] = f"expected string-like, got {df[col].dtype}"
                                split_ok = False
                    if dtype_issues:
                        split_rep["dtype_issues"] = dtype_issues

                split_rep["ok"] = split_ok
                report["splits"][name] = split_rep
                if not split_ok:
                    report["ok"] = False

            if report["ok"]:
                logging.info("Both train and test match schema.")
            else:
                logging.error("Train or test failed schema validation. See report.")

            return report

        except Exception as e:
            logging.error(f"Exception during validation: {e}")
            raise MyException(e, sys)

    def save_validation_report(self, report: dict, filename: str = "validation_report.yaml") -> str:
        """
        Saves the validation report as YAML under artifacts/<timestamp>/data_validation/.
        Returns the path.
        """
        try:
            validation_dir = self.artifact_dir / "data_validation"
            validation_dir.mkdir(parents=True, exist_ok=True)
            report_path = validation_dir / filename
            with open(report_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(report, f, sort_keys=False)
            logging.info(f"Validation report saved at: {report_path}")
            return str(report_path)
        except Exception as e:
            logging.error(f"Error saving validation report: {e}")
            raise MyException(e, sys)

    def run(self) -> tuple[bool, str, dict]:
        """
        Runs validation and writes report.
        Returns: (ok, report_path, report_dict)
        """
        self.prepare_data_validation()
        report = self.validate_number_of_columns()
        path = self.save_validation_report(report)
        return report.get("ok", False), path, report
