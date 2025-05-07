import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

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

@app.get("/moneyplanet-ai")
def hello():
    return "Hello World!"


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000)