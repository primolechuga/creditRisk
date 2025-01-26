from fastapi import APIRouter, Request, status
from pydantic import BaseModel
from app.services.loan_service import LoanService


class RiskProfile(BaseModel):
    profile_vector: str


class LoanController:

    def __init__(self):
        self.router = APIRouter(prefix="/api")
        self.loan_service = LoanService()
        
        self.router.add_api_route("/get-score", self.get_score, methods=["POST"])
        

    async def get_score(self, request: Request):
        
        data = await request.json()
        print(data)

        ret_data = self.loan_service.get_score(data)
        print(ret_data)
        return ret_data


