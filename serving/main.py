from fastapi import FastAPI
from serving.router import router

app = FastAPI(title="OptiCargo ML Models Service", version="1.0")
app.include_router(router)