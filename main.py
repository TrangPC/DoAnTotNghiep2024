import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import math
from typing import List
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import os
from k8s_client import scale_deployment, get_pod_count
from prometheus_client import get_requests_per_minute
from apscheduler.schedulers.background import BackgroundScheduler
import requests
from datetime import datetime

load_dotenv()
SCALER = int(os.getenv("MINMAX_SCALER", 229426)) 
WINDOWSIZE = int(os.getenv("WINDOWSIZE", 10))
PODS_MIN = int(os.getenv("PODS_MIN", 10))
RRS = float(os.getenv("RRS", 0.6)) 
WORKLOAD_POD = int(os.getenv("WORKLOAD_POD", 300))
prometheus_url = os.getenv("PROMETHEUS_URL","http://10.148.0.2:30090")
ingress_name =  os.getenv("APP_INGRESS","helloword-helloworld")
namespace = os.getenv("NAMESPACE_SCALING","app")
deployment= os.getenv("DEPLOYMENT_SCALING","helloword-helloworld")
model = load_model("BiLSTM_autoscaling_ep20.keras")

app = FastAPI()
scheduler = BackgroundScheduler()

# Define the request schema
class PredictionRequest(BaseModel):
    previous_workload: List[float]
    current_pods: int

def make_prediction(model, previous_workload, current_pods):
    """
    Make a prediction from a given input array using a trained model.
    """
    # Normalize the input data
    previous_workload_scaled = previous_workload / SCALER

    # Create a sequence for prediction (sequence_length-sized window)
    input_sequence = []
    input_sequence.append(previous_workload_scaled[-WINDOWSIZE:])

    input_sequence = np.array(input_sequence)

    # Reshape for LSTM (samples, timesteps, features)
    input_sequence = input_sequence.reshape(
        (input_sequence.shape[0], input_sequence.shape[1], 1)
    )

    # Make predictions
    prediction_scaled = model.predict(input_sequence)

    # Reverse the scaling (denormalize the prediction)
    predicted_workload = prediction_scaled * SCALER

    predicted_pods = pods_adaption(predicted_workload, current_pods)

    return predicted_workload, predicted_pods

def pods_adaption(predicted_workload, current_pods):
    pods_t1 = predicted_workload / WORKLOAD_POD

    if pods_t1 > current_pods:
        return math.ceil(pods_t1)

    elif pods_t1 < current_pods:
        pods_t1 = max(pods_t1, PODS_MIN)
        pods_surplus = (current_pods - pods_t1) * RRS
        return math.ceil(max((current_pods - pods_surplus), PODS_MIN))


def scheduled_prediction():
    """
    Function to be executed every minute by the scheduler
    """
    print(f"Prometheus: '{prometheus_url}', Ingress Name: '{ingress_name}'")
    workload_history = get_requests_per_minute(prometheus_url,ingress_name)
    print(f"Workload History: '{workload_history}'")
    try:
        # Ensure we have enough historical data
        if len(workload_history) >= WINDOWSIZE:
            # Get current number of pods from k8s 
            current_pods = get_pod_count(namespace, deployment)  
            
            previous_workload = np.array(workload_history[-WINDOWSIZE:])
            predicted_workload, predicted_pods = make_prediction(
                model=model,
                previous_workload=previous_workload,
                current_pods=current_pods
            )
            
            # Scale the deployment
            scale_deployment(namespace, deployment, predicted_pods)
            
            print(f"[{datetime.now()}] Scheduled prediction executed: workload={predicted_workload}, pods={predicted_pods}")
    except Exception as e:
        print(f"Error in scheduled prediction: {str(e)}")


@app.post("/api/predict")
def predict(request: PredictionRequest):
    if len(request.previous_workload) < WINDOWSIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Input array must be at least {WINDOWSIZE}",
        )

    # # Update workload history
    # workload_history.extend(request.previous_workload)
    # # Keep only the most recent data points
    # while len(workload_history) > WINDOWSIZE * 2:
    #     workload_history.pop(0)

    predicted_workload, predicted_pods = make_prediction(
        model=model,
        previous_workload=np.array(request.previous_workload),
        current_pods=request.current_pods,
    )

    scale_deployment("app", "helloword-helloworld", predicted_pods)
    return JSONResponse(
        status_code=200,
        content={
            "predicted_workload": float(predicted_workload),
            "predicted_pods": predicted_pods,
        },
    )

@app.on_event("startup")
def start_scheduler():
    scheduler.add_job(scheduled_prediction, 'interval', minutes=1)
    scheduler.start()

@app.on_event("shutdown")
def shutdown_scheduler():
    scheduler.shutdown()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)