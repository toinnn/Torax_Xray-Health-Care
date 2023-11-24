from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

app = FastAPI()


@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.get("/ola")
async def hello_():
    return "Olá codigo"

# Configurando o manipulador para caminhos relativos
@app.get("/{file_path:path}")
async def read_file(file_path: str, request: Request):
    file_path = Path(file_path)

    # Verificando se o arquivo existe
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")

    return FileResponse(file_path, media_type="text/html" if file_path.suffix == ".html" else None)

# Configurando o diretório de templates para o Jinja2
templates = Jinja2Templates(directory="static")