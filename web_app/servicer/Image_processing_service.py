import json
# from torchvision.io import decode_png, read_image , ImageReadMode
# import torchvision.transforms as T
import sys
import os
import pickle
import joblib

import torch
# import torch.nn as nn
# from Transformer_Decoder import decoder
# import torchvision.transforms as T

# Obtém o caminho absoluto do diretório avô do diretório deste arquivo :
diretorio_pai = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
diretorio_avo = os.path.abspath(os.path.join(diretorio_pai, ".."))

# Adiciona o diretório do projeto ao sys.path :
sys.path.append( diretorio_avo )
# from my_models import my_model



class image_processing_service() :

    def __init__(self , model = None , dict = None) -> None :
        self.model = model
        self.dictionary = dict

    def load_model(self, path) -> None :
        with open(path, 'rb') as file:
            model = pickle.load(file)
            # self.model = joblib.load(file)
        self.model = model
        return model
    
    def use_model(self , input):
        out = self.model(input)
        return [ self.dictionary[i] for i in out]

# meu = my_model()
         


