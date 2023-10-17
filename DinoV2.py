import torch
import torch.nn as nn
from Transformer_Decoder import decoder , Trainer
import json
from torchvision.io.image import decode_png, read_file
import polars as pl
import os

def load_image_nvjpngl_gpu(image_path):
    """
    Loads an image from the specified file path using NVJPEG decoder on GPU and returns a PyTorch tensor.

    Args:
        image_path (str): The path to the image file.

    Returns:
        torch.Tensor: The PyTorch tensor representing the image.
    """
    data = read_file(image_path)
    tensor = decode_png(data).to("cuda")
    return tensor



# vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
# vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')


# features = vits16(your_input)
print("Rodou com sucesso")

path = "D://Bibliotecas//Documents//Universitário//UFC//Tópicos Avançados-Visão Computacional do Vitor Hugo//Dataset NIH-Chest-X-Ray//DataSet__NIH-Chest-X-Ray___archive//"
train_path = path + "train_val_list.txt"
test_path  = path + "test_list.txt"
data_intro_path = path + "Data_Entry_2017.csv"
image_dir_path = path + "images_001//images"


train_id = open(train_path , "r").readline()



image = load_image_nvjpngl_gpu(image_dir_path + "//00000001_000.png")
data_intro = pl.read_csv(data_intro_path)

classes  ="1, Atelectasis; 2, Cardiomegaly; 3, Effusion; 4, Infiltration; 5, Mass; 6, Nodule; 7, Pneumonia; 8,Pneumothorax; " 
classes += "9, Consolidation; 10, Edema; 11, Emphysema; 12, Fibrosis; 13, Pleural_Thickening; 14 Hernia"
print(train_id)
print(classes)
# print(image)
# data_intro = 
# data_intro = [row for row in data_intro[['Image Index' ,'Finding Labels' ]].head(1).iter_rows(named=True) ]
# print(data_intro[0]['Image Index'].replace(".png" , ""))
class2label = json.load(open("C://Users//limaa//Dataset_work_space//Torax_Xray//target_dict_class2label.json" , "rb"))
label2class = json.load(open("C://Users//limaa//Dataset_work_space//Torax_Xray//target_dict_label2class.json" , "rb"))
print(label2class)

path_2_target_tabel = "C://Users//limaa//Dataset_work_space//Torax_Xray//target_tabel.arrow"
"""target = []
for row in data_intro[['Image Index' ,'Finding Labels' ]].iter_rows(named=True) :
    target += [ {'Image Index' : row['Image Index'] , 'Finding Labels' : [ int(label2class[i]) for i in row['Finding Labels'].split("|")] } ]
target = pl.DataFrame(target)
# print(target)

target.write_ipc(path_2_target_tabel)"""
# target = pl.read_ipc(path_2_target_tabel)
# print(target)

# for i in pl.DataFrame(target).iter_rows():
#     print(i)

ark = os.listdir(image_dir_path)
print(ark)

class my_model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(my_model).__init__(*args, **kwargs)
        self.encoder = vits14 #EU NÃO LEMBRO QUANTO DEVERIA SER O MODEL_DIM !!!!!
        self.decoder = decoder(model_dim = 176,heads = 8 ,num_layers = 8 , num_Classes = 14 , device = torch.device("cuda"))

        def forward_fit(self, image  , seq_max_lengh = 100):
            enc = self.encoder(image)
            return self.decoder.forward_fit(enc , enc , seq_max_lengh)
        
        def forward(self, image  , seq_max_lengh = 100):
            enc = self.encoder(image)
            return self.decoder(enc , enc , seq_max_lengh)
        
