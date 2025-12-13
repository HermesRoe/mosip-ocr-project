import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # <--- NEW IMPORT

# FIX IMPORT PATH: Add the parent directory so we can find 'Opencv' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.extract import ExtractRouter
from api.verify import VerifyRouter

app = FastAPI()

# --- CRITICAL FIX FOR FRONTEND ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Allows your React App
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ExtractRouter)
app.include_router(VerifyRouter)

@app.get("/") # Changed from /app to / for easier testing
def root():
    return {"message": "MOSIP OCR Backend is Running"}
