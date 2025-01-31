from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


class CORS:
    @staticmethod
    def add(app: FastAPI):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://creditriskfront.onrender.com"],        
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )

    