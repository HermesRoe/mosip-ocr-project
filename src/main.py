from fastapi import FastAPI

from api.extract import ExtractRouter
from api.verify import VerifyRouter


app = FastAPI()


app.include_router(ExtractRouter)
app.include_router(VerifyRouter)


@app.get("/app")
def root():
    return {"message": "root"}
