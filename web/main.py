from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager
from .model.model import Predictor
from .model.preprocessor import Preprocessor
import torch
from .schema import PredictionBody, HealthCheck
from .config import CONFIG
import numpy as np

@asynccontextmanager
async def lifetime_event(app: FastAPI):
    """
    Initialize FastAPI and add variables
    """
    preprocessor = Preprocessor(lookback_window=10, window_length=5, polyorder=3)
    
    # Initialize the pytorch model
    model = Predictor(input_size=16, hidden_size=128, output_size=2, kernel_size=2)
    model.load_state_dict(torch.load(CONFIG['MODEL_PATH']))
    model.eval()

    # add model and other preprocess tools too app state
    app.package = {
        "model": model,
        "preprocessor": preprocessor,
    }
    yield

app = FastAPI(
    lifespan=lifetime_event,
    title="Forecaster CNN-LSTM-TL",
    description="An API for making predictions of CPU utilization and memory utilization.",
    openapi_tags=[
        {
            "name": "healthcheck",
            "description": "Endpoints for performing health checks on the API."
        },
        {
            "name": "prediction",
            "description": "Endpoints for making predictions using the machine learning model."
        },
        {
            "name": "model-management",
            "description": "Endpoints for managing and monitoring the machine learning model."
        }
    ],           
)

@app.get("/")
async def home():
    return {"message": "CSL Predictor"}

@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
    response_description="Return HTTP Status Code 200 (OK)",
)
def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")

@app.post(
    '/predict/',
    summary='Predicts future CPU and memory loads based on the provided initial data using a recursive prediction approach',
    status_code=status.HTTP_200_OK,
    response_model=PredictionBody,
    response_description="Return HTTP Status Code 200 (OK)",
    tags=['prediction']
)
async def predict(data: PredictionBody):
    """
    ## Predicting CPU and Memory Loads (6 min / 2 min interval)
    Predicts future CPU and memory loads based on the provided initial data using a recursive prediction approach.

    Args:
        data (PredictionBody): An object containing the initial CPU and memory load values. 
                               It should include `cpu_loads` and `memory_loads`, which are arrays of historical data 
                               used as input for the prediction model. Each element must be a value in range of 
                               0 - 1 and it's assumed the value is mean of 2 min intervals. A minimum length of 15 is required.

    Raises:
        HTTPException: Raised if an error occurs during the prediction process, such as issues with model loading, 
                       preprocessing, or inference. The exception will return a 500 status code with the error details.

    Returns:
        dict: A dictionary containing the predictions for the next 3 time steps. Each prediction includes:
              - "cpu": The predicted CPU load, rounded to the number of digits specified in the configuration.
              - "memory": The predicted memory load, rounded to the number of digits specified in the configuration.
    """
    try:
        # Load model and preprocessor
        model = app.package['model']
        preprocessor: Preprocessor = app.package['preprocessor']
        
        # Initial data (first 5 rows)
        cpu_loads = data.cpu_loads
        memory_loads = data.memory_loads
        
        if len(cpu_loads) < 15 and len(memory_loads) < 15:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A minimum length of 15 is required")
        
        # Recursive prediction loop
        predictions = []
        for _ in range(3):  # Predict 3 steps ahead
            # Preprocess and get last sequence
            inputs = preprocessor.transform(cpu_loads, memory_loads)  # Shape: (1, 5, 16)
            
            model.eval()
            print(inputs)
            with torch.inference_mode():
                output = model(inputs.permute(0, 2, 1)).squeeze().tolist() 
            
            # Append prediction
            predictions.append(output)
            
            # Update inputs for next step
            cpu_loads = np.append(cpu_loads[1:], output[0])  # Shift CPU loads and add new prediction
            memory_loads = np.append(memory_loads[1:], output[1])  # Shift memory loads and add new prediction
        
        # Return all predictions
        return {
            "predictions": [
                {"cpu": round(pred[0], CONFIG['ROUND_DIGIT']), "memory": round(pred[1], CONFIG['ROUND_DIGIT'])} for pred in predictions
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    