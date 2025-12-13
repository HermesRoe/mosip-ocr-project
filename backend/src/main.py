import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- PATH FIX START ---
# We force Python to look 2 levels up (src -> backend -> ROOT)
# This allows 'api.extract' to find the 'Opencv' folder imports.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# --- PATH FIX END ---

from api.extract import ExtractRouter
from api.verify import VerifyRouter

app = FastAPI()

# Enable CORS so Frontend (Port 3000) can talk to Backend (Port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ExtractRouter)
app.include_router(VerifyRouter)

@app.get("/")
def root():
    return {"message": "MOSIP OCR Backend is Running"}