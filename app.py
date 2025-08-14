from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
from pipeline.prediction_pipeline import PredictionPipeline
from pipeline.training_pipeline import TrainingPipeline
from constants import APP_HOST, APP_PORT
from uvicorn import run as app_run

app = FastAPI()
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

@app.get("/")
def index(request: Request):
	return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about")
def about(request: Request):
	return templates.TemplateResponse("about.html", {"request": request})

@app.get("/train")
def train(request: Request):
	try:
		pipeline = TrainingPipeline()
		pipeline.run()
		return templates.TemplateResponse("train.html", {"request": request, "message": "Training completed successfully."})
	except Exception as e:
		return templates.TemplateResponse("train.html", {"request": request, "message": f"Training failed: {e}"})

@app.get("/predict")
def predict_page(request: Request):
	return templates.TemplateResponse("predict.html", {"request": request})

@app.post("/predict")
async def predict(request: Request):
	try:
		data = await request.json()
		df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
		pipeline = PredictionPipeline()
		preds = pipeline.predict_from_df(df)
		return JSONResponse(content={"predictions": preds})
	except Exception as e:
		return JSONResponse(content={"error": str(e)}, status_code=500)



if __name__ == "__main__":
	app_run(app, host=APP_HOST, port=APP_PORT)