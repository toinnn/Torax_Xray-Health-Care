from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

import json
# from torchvision.io import decode_png, read_image , ImageReadMode
# import torchvision.transforms as T
import sys
import os
import io
import pickle
# import joblib

import torch
# import torch.nn as nn
# from Transformer_Decoder import decoder
# import torchvision.transforms as T

# Obtém o caminho absoluto do diretório avô do diretório deste arquivo :
# diretorio_pai = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# diretorio_avo = os.path.abspath(os.path.join(diretorio_pai, ".."))

# Adiciona o diretório do projeto ao sys.path :
# sys.path.append( diretorio_avo )
# sys.path.append( diretorio_pai )

from servicer.my_models import my_model , my_model_medium , my_model_small
# from servicer.Transformer_Decoder import decoder



class image_processing_service() :
    path  = "I.A._Models//xRay_model_0_loss_0.03098666666666666.model"
    path1 = "I.A._Models//xRay_model_1_loss_0.03098666666666666.model"
    path_1 = "I.A._Models//state__xRay_model_1_loss_0.03098666666666666.model"
    path_2 = "I.A._Models//state__xRay_model_2_loss_0.029229629629629626.model"
    path_3 = "I.A._Models//state__xRay_model_3_loss_0.029611111111111116.model"
    path_4 = "I.A._Models//state__xRay_model_4_loss_0.06855873015873015.model"
    path_5 = "I.A._Models//state__xRay_model_5_loss_0.03429714285714285.model"
    path_6 = "I.A._Models//state__xRay_model_6_loss_0.03428571428571429.model"
    path_7 = "I.A._Models//state__xRay_model_7_loss_0.03432952380952381.model"
    path_8 = "I.A._Models//state__xRay_model_8_loss_0.034127619047619046.model"
    path_9 = "I.A._Models//state__xRay_model_9_loss_0.034420952380952385.model"

    def __init__(self , model_class = my_model_small , model_args : dict = {} ,model_path = path_9 , dict = None) -> None :
        self.model = None
        # self.load_model(model_path)
        self.load_model(model_path , model_class , model_args )
        self.dictionary = dict

    # def load_model(self, path) -> None :
    #     with open(path, 'rb') as file:
    #         # model = pickle.load(file)
    #         self.model = pickle.load(file)
    #     # self.model = model
    #     # return model

    def load_model(self, path , model_class , model_args : dict ) :
        instance = model_class(**model_args)
        instance.load_state_dict(torch.load(path))
        self.model =  instance
    
    def use_model(self , input):
        out = self.model(input)
        return [ self.dictionary[i] for i in out]
    
image_model = image_processing_service()

if torch.cuda.is_available():
    image_model.model.setDevice(torch.device("cuda"))
def pega_argmax(tensor_input) :
    out = torch.argmax(tensor_input , dim = 2)
    return out
def show_resuts(contents):
    
    # path_model_0 = "I.A._Models//xRay_model_0_loss_0.03098666666666666.model"
    # path_model_1 = "I.A._Models//xRay_model_1_loss_0.03098666666666666.model"


    # if __name__ == "__main__":
    # image_model.load_model( path_model_1 )
    
    # try:
    # Aqui você pode adicionar lógica adicional para processar o arquivo se necessário
    print("Entrou aqui no show resuts")
    # Lê os bytes da imagem
    # contents = await file.read()

    # Converte os bytes para um objeto de imagem usando o Pillow
    image = Image.open(io.BytesIO(contents))
    image_tensor = torch.tensor([np.array(image)]).float()
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    # image_tensor = image_tensor.view(1 ,image_tensor.shape[0] , image_tensor.shape[1] )
    print(image_model , image_tensor.shape )
    # print(f"\n\n\nResposta do modelo = {image_model.model }\n\n")

    # out = image_model.model.forward_fit(image_tensor , max_lengh = 9 )
    # out = pega_argmax(out)

    out = image_model.model(  image_tensor  )
    print(f"\n\n\nResposta do modelo = { out }\n\n")
    # print(io.BytesIO(contents))
    # print(np.array(image).shape)

    # # Converte a imagem para um array NumPy
    # image_array = np.array(image)

    # # Crie uma representação visual da imagem usando matplotlib
    # plt.imshow(image_array)
    # plt.axis('off')  # Desliga os eixos
    # plt.tight_layout(pad=0)

    # plt.show()
    # plt.close()

    # return out

    
# meu = my_model()
# ia_test = image_processing_service()
         


