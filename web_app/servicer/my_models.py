import torch
import torch.nn as nn
from servicer.Transformer_Decoder import decoder
import torchvision.transforms as T


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

class my_model_medium(nn.Module):
    def __init__(self ,  device : torch.device = torch.device("cpu")) -> None:
        super(my_model_medium , self  ).__init__()
        vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # vits14.eval()
        # print(vits14.eval())
        self.encoder = vits14.to(device) #EU NÃO LEMBRO QUANTO DEVERIA SER O MODEL_DIM !!!!!
        self.decoder = decoder(model_dim = 768 ,heads = 4 ,num_layers = 5 , num_Classes = 16 , device = device)
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

class my_model_small(nn.Module):
    def __init__(self ,  device : torch.device = torch.device("cpu")) -> None:
        super(my_model_small , self  ).__init__()
        vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # vits14.eval()
        # print(vits14.eval())
        self.encoder = vits14.to(device) #EU NÃO LEMBRO QUANTO DEVERIA SER O MODEL_DIM !!!!!
        self.decoder = decoder(model_dim = 768 ,heads = 3 ,num_layers = 3 , num_Classes = 16 , device = device)
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
# my = my_model()