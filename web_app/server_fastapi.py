from fastapi import FastAPI, Request, HTTPException , File , UploadFile
from fastapi.responses import FileResponse
# from fastapi.templating import Jinja2Templates
import uvicorn
from pathlib import Path
import io
import os
import sys
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from servicer.Image_processing_service import show_resuts #image_processing_service


# vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

# Obtém o caminho absoluto do diretório avô do diretório deste arquivo :
diretorio_pai = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Adiciona o diretório do projeto ao sys.path :
sys.path.append( diretorio_pai )
from servicer.my_models import * #my_model 


# image_model = image_processing_service()

# path_model_0 = "D://Bibliotecas//Documents//Dev_Projects//PythonProjects//VsCodePython//UFC//BioData//Torax_Xray-Health-Care//I.A._Models//xRay_model_0_loss_0.03098666666666666.model"
# path_model_1 = "D://Bibliotecas//Documents//Dev_Projects//PythonProjects//VsCodePython//UFC//BioData//Torax_Xray-Health-Care//I.A._Models//xRay_model_1_loss_0.03098666666666666.model"


# # if __name__ == "__main__":
#     image_model.load_model( path_model_1 )
app = FastAPI()


@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.get("/ola")
async def hello_():
    return "Olá codigo"

@app.post("/send_image")
async def send_image(file: UploadFile = File(...)):
    
    # image_model = image_processing_service()

    # path_model_0 = "I.A._Models//xRay_model_0_loss_0.03098666666666666.model"
    # path_model_1 = "I.A._Models//xRay_model_1_loss_0.03098666666666666.model"


    # if __name__ == "__main__":
    # image_model.load_model( path_model_1 )
    
    try:
        # Aqui você pode adicionar lógica adicional para processar o arquivo se necessário
        print("Entrou aqui no try")
        # Lê os bytes da imagem
        contents = await file.read()
        show_resuts(contents)
        # out = show_resuts(contents)
        
        # # Converte os bytes para um objeto de imagem usando o Pillow
        # image = Image.open(io.BytesIO(contents))
        # print(image_model , torch.tensor(np.array(image)) )
        # print(f"\n\n\nResposta do modelo = {image_model.model }\n\n")
        # # print(f"\n\n\nResposta do modelo = {image_model.model(torch.tensor(np.array(image))) }\n\n")
        # # print(io.BytesIO(contents))
        # print(np.array(image).shape)

        # # Converte a imagem para um array NumPy
        # image_array = np.array(image)

        # # Crie uma representação visual da imagem usando matplotlib
        # plt.imshow(image_array)
        # plt.axis('off')  # Desliga os eixos
        # plt.tight_layout(pad=0)

        # plt.show()
        # plt.close()

        # print("saíiii")
        # resposta_json = {"mensagem": "Arquivo recebido com sucesso!", "status": "OK"}
        # return resposta_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Configurando o manipulador para caminhos relativos
@app.get("/{file_path:path}")
async def read_file(file_path: str, request: Request):
    file_path = Path(file_path)

    # Verificando se o arquivo existe
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    

    return FileResponse(file_path, media_type="text/html" if file_path.suffix == ".html" else None)

# # Configurando o diretório de templates para o Jinja2
# templates = Jinja2Templates(directory="static")


if __name__ == "__main__":
    

    uvicorn.run("server_fastapi:app", workers=1, host='0.0.0.0', port=8000, reload=True)