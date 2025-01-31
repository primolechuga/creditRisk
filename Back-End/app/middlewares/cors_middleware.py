from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


class CORS:
    @staticmethod
    def add(app: FastAPI, allowed_origins: list):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,        
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )

    