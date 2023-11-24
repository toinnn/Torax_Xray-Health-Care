from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()


@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.get("/ola")
async def hello_():
    return "Olá codigo"