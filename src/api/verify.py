from fastapi import APIRouter
from models import VerificationRequest


VerifyRouter = APIRouter(tags=["verify"])


@VerifyRouter.post("/verify")
def verify_data(request: VerificationRequest):
    ...
