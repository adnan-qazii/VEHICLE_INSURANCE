#test logging and exception handling
from logger import logging
from exception import MyException
import sys



num1=5
num2=0
logging.info("Starting division operation")
try:
    result = num1 / num2
    logging.info(f"Division result: {result}")
except Exception as e:
    raise MyException("Error occurred", sys) from e
