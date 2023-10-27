import torch
import torch.nn as nn
from Transformer_Decoder import decoder , Trainer
from torch.utils.data import DataLoader , Dataset
import json
# from torchvision.io.image import decode_png, read_image
from torchvision.io import decode_png, read_image , ImageReadMode
import torchvision.transforms as T
import polars as pl
import os
# os.environ['MP_METHOD'] = 'spawn'

def load_image_nvjpngl_gpu(image_path):
    """
    Loads an image from the specified file path using NVJPEG decoder on GPU and returns a PyTorch tensor.

    Args:
        image_path (str): The path to the image file.

    Returns:
        torch.Tensor: The PyTorch tensor representing the image.
    """
    data = read_image(image_path , mode = ImageReadMode.GRAY )
    return data.float()
    # print(data)
    tensor = decode_png(data).float()#.to("cuda")
    return tensor


class dataset_NIH_Chest(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs : list[str] , Data_Entry_path : str , image_dir_path : str , label2class_path : str , max_label_lengh : int ):
        'Initialization'
        # self.labels = labels
        self.list_IDs = tuple(list_IDs) #Lista contendo os nomes das imagens
        self.image_dir_path = image_dir_path #string com o path da pasta que contem as imagens de input
        self.Data_Entry  = pl.read_csv(Data_Entry_path)
        self.label2class = json.load(open(label2class_path , "rb"))
        self.y_len_max = max_label_lengh + 1
        # for row in data_intro[['Image Index' ,'Finding Labels' ]].iter_rows(named=True) :
        #     print({row['Image Index'] : row['Finding Labels'].split("|") })
        #     raise
        self.Data_Entry = {row['Image Index'] : row['Finding Labels'].split("|") for row in self.Data_Entry[['Image Index' ,'Finding Labels' ]].iter_rows(named=True)}
        """for row in data_intro[['Image Index' ,'Finding Labels' ]].iter_rows(named=True) :
            if max_label_lengh < len( row['Finding Labels'].split("|")  )  : """
        keys = self.Data_Entry.keys()
        self.name_2_index = {self.Data_Entry[i]: i+1  for i in range(keys) }

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample  
        ID = self.list_IDs[index]
        # linha_encontrada = self.Data_Entry.filter(( self.Data_Entry['Image Index'] == self.list_IDs[index] ) )
        # linha_encontrada['Finding Labels'].split("|")
        
        y = torch.tensor([ int(self.label2class[i]) for i in self.Data_Entry[ID]] ).view(-1, 1) #linha_encontrada['Finding Labels'][0].split("|") ]).view(-1, 1)
        
        # aux = self.y_len_max - y.shape[0]
        # if aux != 0 :
        aux = torch.zeros( self.y_len_max - y.shape[0]  , 1)
        y = torch.cat([y , aux] , dim = 0)

        # Load data and get label
        X = load_image_nvjpngl_gpu(self.image_dir_path + self.list_IDs[index]) #torch.load( self.image_dir_path + self.list_IDs[index] )
        # y = self.labels[ID]

        return X, y


if __name__ == '__main__': 

    

    
    # vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    # vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')


    # features = vits16(your_input)
    print("Rodou com sucesso")

    path = "D://Bibliotecas//Documents//Universitário//UFC//Tópicos Avançados-Visão Computacional do Vitor Hugo//Dataset NIH-Chest-X-Ray//DataSet__NIH-Chest-X-Ray___archive//"
    
    train_path = path + "train_val_list.txt"
    test_path  = path + "test_list.txt"
    data_intro_path = path + "Data_Entry_2017.csv"
    image_dir_path = "C://Users//limaa//Dataset_work_space//Torax_Xray//images"
    label2class_path = "C://Users//limaa//Dataset_work_space//Torax_Xray//target_dict_label2class.json"


    train_id = [i.replace("\n" , "") for i in open(train_path , "r").readlines() ]
    test_id  = [i.replace("\n" , "") for i in open(test_path , "r" ).readlines() ]


    image = load_image_nvjpngl_gpu(image_dir_path + "//00000001_000.png")
    data_intro = pl.read_csv(data_intro_path)


    print(train_id[-2])
    print(test_id[-2])
    print(data_intro.height)

    linha_encontrada = data_intro.filter(( data_intro['Image Index'] == "00000001_002.png" ) )
    print(linha_encontrada['Finding Labels'][0].split("|"))

    max_label_lengh = 0
    for row in data_intro[['Image Index' ,'Finding Labels' ]].iter_rows(named=True) :
        if max_label_lengh < len( row['Finding Labels'].split("|")  )  : 
            max_label_lengh = len( row['Finding Labels'].split("|") ) 
            print(f"a maior label é de tamanho : {max_label_lengh}" )




    params = {'batch_size': 8 ,
        'shuffle': True,
        'num_workers': 6}
        


    training_set = dataset_NIH_Chest(train_id , data_intro_path , image_dir_path + "//" , label2class_path , max_label_lengh )
    training_loader = DataLoader(training_set, **params)

    test_set = dataset_NIH_Chest(test_id , data_intro_path , image_dir_path + "//" , label2class_path , max_label_lengh )
    test_loader = DataLoader(test_set, **params)

    print(training_set.y_len_max)



    """dataloader : DataLoader ,n , maxErro , maxAge = 1 ,mini_batch_size = 1  , #input_Batch :list , target_Batch : list, n , maxErro , maxAge = 1 ,mini_batch_size = 1  ,
                lossFunction = nn.CrossEntropyLoss() ,lossGraphPath = None , test_dataloader = None ,#Input_Batch = None,
                out_max_Len  = 150
    """
    # data_intro = 
    # data_intro = [row for row in data_intro[['Image Index' ,'Finding Labels' ]].head(1).iter_rows(named=True) ]
    # print(data_intro[0]['Image Index'].replace(".png" , ""))
    class2label = json.load(open("C://Users//limaa//Dataset_work_space//Torax_Xray//target_dict_class2label.json" , "rb"))
    label2class = json.load(open("C://Users//limaa//Dataset_work_space//Torax_Xray//target_dict_label2class.json" , "rb"))
    # print(label2class)

    path_2_target_tabel = "C://Users//limaa//Dataset_work_space//Torax_Xray//target_tabel.arrow"
    path_2_input_tabel = "C://Users//limaa//Dataset_work_space//Torax_Xray//input//input_tabel_"
    path_2_counter = "C://Users//limaa//Dataset_work_space//Torax_Xray//counter.json"

    with open(path_2_counter , "r") as arquivo_json:
        counter = json.load(arquivo_json)

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

    """
    ark = os.listdir(image_dir_path)
    # print(ark)
    input = []
    ctd = counter["ctd_generic"]
    for img_name in ark[counter["ctd"]*100: ] :
        img = load_image_nvjpngl_gpu( image_dir_path +"//" + img_name )
        input += [{'Image Index' : img_name , "image_tensor" : img.tolist() }]
        ctd += 1
        if ctd % 100 == 0 :
            print(ctd)
            input = pl.DataFrame(input)
            print(input)
            input.write_ipc(path_2_input_tabel + str(counter["ctd"]) + ".arrow")
            counter["ctd"] += 1
            counter["ctd_generic"] = ctd
            with open(path_2_counter , "w") as arquivo_json:
                json.dump(counter , arquivo_json)
            del input
            input = []
    input = pl.DataFrame(input)
    print(input)
    input.write_ipc(path_2_input_tabel + str(counter["ctd"]) + ".arrow")
    counter["ctd"] += 1
    with open(path_2_counter , "w") as arquivo_json:
        json.dump(counter , arquivo_json)
    del input
    input = []
    """

    name_id = "00003890_004.png"
    """#ESSA VERSÃO NÃO SE JUSTIFICA
    input = pl.read_ipc(path_2_input_tabel + str(counter["ctd"] - 1) + ".arrow")
    print( torch.tensor(input.filter((input["Image Index"] == name_id ))["image_tensor"].to_list() ).cuda().shape )
    """
    print(load_image_nvjpngl_gpu("C://Users//limaa//Dataset_work_space//Torax_Xray//images//" + name_id ).cuda().shape )


    class my_model(nn.Module):
        def __init__(self ,  device : torch.device = torch.device("cpu")) -> None:
            super(my_model , self  ).__init__()
            vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            # vits14.eval()
            # print(vits14.eval())
            self.encoder = vits14.to(device) #EU NÃO LEMBRO QUANTO DEVERIA SER O MODEL_DIM !!!!!
            self.decoder = decoder(model_dim = 768 ,heads = 8 ,num_layers = 8 , num_Classes = 16 , device = device)
            self.device  = device
            # self.transform_image = T.Compose([T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])] )
            self.transform_image = T.Compose([T.Resize(224),  T.Normalize([0.5], [0.5])] )

        def setDevice(self , device : torch.device) :
            self.encoder = self.encoder.to( device )
            self.decoder.setDevice( device )
            return


        def forward_fit(self, image  , max_lengh = 100):
            # print(f"Passou do Encoder image : { image.shape}")
            img2 = []
            for img in image :
                img = img.view(1, img.shape[0] , img.shape[1])
                img = torch.cat( [img,img,img] , dim = 0 )
                # print(f"img.shape = {img.shape}")
                # print(f"trasnform : {self.transform_image( img  )[:3].unsqueeze(0).shape }")
                img2 += [self.transform_image( img )[:3].unsqueeze(0) ]
            # image = [ self.encoder(self.transform_image( img.view(1 , img.shape[0] , img.shape[1]))[:3].unsqueeze(0) ).view(1,1,-1)     for img in image]
            image = [ self.encoder(img).view(1 , 1 , -1) for img in img2 ]
            enc   = torch.cat(image , dim = 0 ) 
            # image = self.transform_image(image)[:3].unsqueeze(0) 
            # print(f"Passou do Encoder image : {image.shape}")
            # enc = self.encoder(image)
            # print(f"Passou do Encoder enc : {enc.shape}")
            return self.decoder.forward_fit(enc , enc , max_lengh)
        
        def forward(self, image  , max_lengh = 100):
            # image = self.transform_image(image)[:3].unsqueeze(0) 
            # enc = self.encoder(image)

            img2 = []
            for img in image :
                img = img.view(1, img.shape[0] , img.shape[1])
                img = torch.cat( [img,img,img] , dim = 0 )
                # print(f"img.shape = {img.shape}")
                # print(f"trasnform : {self.transform_image( img  )[:3].unsqueeze(0).shape }")
                img2 += [self.transform_image( img )[:3].unsqueeze(0) ]
            # image = [ self.encoder(self.transform_image( img.view(1 , img.shape[0] , img.shape[1]))[:3].unsqueeze(0) ).view(1,1,-1)     for img in image]
            image = [ self.encoder(img).view(1 , 1 , -1) for img in img2 ]
            enc   = torch.cat(image , dim = 0 ) 

            return self.decoder(enc , enc , max_lengh)
            
    model   = my_model(torch.device("cuda"))
    trainer = Trainer(model , torch.device("cuda") )
    trainer.fit(training_loader  , 0.05 , 1 , 1 , test_dataloader = test_loader )


    # vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    # img = load_image_nvjpngl_gpu("C://Users//limaa//Dataset_work_space//Torax_Xray//images//00000099_007.png")
    # transform_image = T.Compose([T.Resize( 224)  , T.Normalize([0.5], [0.5])] )
    # img = torch.cat([img , img , img ] , dim = 0)
    # print(img.shape)
    
    # a = transform_image(img)[:3].unsqueeze(0) 
    # print(f"img.shape : {img.shape}  e  transform_image(img)[:3].unsqueeze(0) :  {a.shape}")
    # print(vits14( a ).shape)

