
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()
scheduler = BackgroundScheduler()
def hello():
    print("aaaaa")

@app.on_event("startup")
def start_scheduler():
    scheduler.add_job(hello, 'interval', minutes=1)
    scheduler.start()

@app.on_event("shutdown")
def shutdown_scheduler():
    scheduler.shutdown()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)