import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from app.modules.decision.interface.controller.v1.decision_controller import router as decision_router
from app.containers import Container

app = FastAPI()
app.container = Container()

app.include_router(decision_router)

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000)