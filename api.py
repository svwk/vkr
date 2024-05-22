from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scripts.api.router import api_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
