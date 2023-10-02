import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
import random as rd
from torch.autograd import Variable
# import heapq
# from skip_Gram import skip_gram

def oneHotEncode(dim, idx):
    vector = torch.zeros(dim)
    vector[idx] = 1.0
    return vector    
def mult_oneHotEncode(dim , idx) :
    return torch.cat( [ oneHotEncode(dim , i ).view(1,-1) for i in idx ]  , dim = 0 ).float()

def word2vec(wordVec , word ,dim):
    try:
        
        return torch.from_numpy(wordVec[word]).view(1,-1)
    except :
        print("Entrou no vetor não pertencente ao vocabulario ")
        return torch.ones([1,dim]).float()*-79

def sen2vec(gensimWorVec , sentence , dim) :
    sen = [token for token in re.split(r"(\W)", sentence ) if token != " " and token != "" ]
    print(sen)
    return torch.cat( [ word2vec(gensimWorVec , word ,dim ) for word in sen ] , dim = 0 )
    
def json2vec(js , key ,dim , gensimWorVec ):
    try :
        seq = [ vec for sen in js[key] for vec in (sen2vec(gensimWorVec , sen.lower() , dim ) , torch.ones([1,dim])*-127 ) ]
        seq.pop()
        return torch.cat( seq , dim = 0 )
    except :
        # Key não existente :
        print("Entrou no não existe a key ")
        return torch.ones([1,dim]).float()*-47
            
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device = torch.device("cpu")):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.d_model = d_model
        
        
        
    def setDevice(self, dev):
        self.device = dev
    
        # self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):

        pos_embedding = torch.zeros(x.shape[0] , x.shape[1], self.d_model , device = self.device)

        pos = torch.arange( x.shape[1] ,device = self.device).float().view(-1,1)
        # input_sin =  #Seno
        # input_cos =  #Cosseno

        embed_pos_of_sin = torch.arange(x[: , : , 0::2].shape[-1] )*(2.0)
        embed_pos_of_cos = torch.arange(int(x.shape[-1]/2))*(2.0)

        sin_div_term = torch.exp( embed_pos_of_sin  * -(torch.log(tensor(10000)) / self.d_model)  ).view(1 , -1).to(self.device)
        cos_div_term = torch.exp(embed_pos_of_cos  * -(torch.log(tensor(10000)) / self.d_model) ).view(1 , -1).to(self.device)

        pos_embedding[: , : , 0::2 ] = torch.sin(pos * sin_div_term )
        pos_embedding[: , : , 1::2 ] = torch.cos(pos * cos_div_term )

        return x.to(self.device) + pos_embedding

# att = nn.MultiheadAttention(embed_dim= model_dim , num_heads= heads )

class selfAttention(nn.Module):
    def __init__(self , model_dim ,heads = 1 , device = torch.device("cpu")):
        super(selfAttention , self ).__init__()
        assert model_dim % heads == 0 , "O número de heads deve ser um divisor do número de dimensões do vetor de Embedding "
        self.heads    = heads 
        self.head_dim = int(model_dim / heads)

        self.att = nn.MultiheadAttention(embed_dim= model_dim , num_heads= heads , device=device)
        self.q = nn.Linear( model_dim , model_dim).to(device)
        self.k = nn.Linear( model_dim , model_dim).to(device)
        self.v = nn.Linear( model_dim , model_dim).to(device)
        self.device = device

        """self.keys    = nn.ModuleList([ nn.Linear(model_dim , self.head_dim , bias = False)  for i in range(heads)])
        self.queries = nn.ModuleList([ nn.Linear(model_dim , self.head_dim , bias = False)  for i in range(heads)])
        self.values  = nn.ModuleList([ nn.Linear(model_dim , self.head_dim , bias = False)  for i in range(heads)])
        
        self.v = Variable(torch.rand(heads , self.head_dim , dtype = float) , requires_grad = True).float()
        self.u = Variable(torch.rand(heads , self.head_dim , dtype = float) , requires_grad = True).float()"""
        # self.output  = nn.Linear(model_dim , model_dim )
        print("chegou aqui")

    def setDevice(self , device ) :
        self.q.to(device)
        self.k.to(device)
        self.v.to(device)
        self.att.to(device)
        self.device = device

    def weights(self)->list :
        weights = [self.output ]
        weights += [i for i in self.keys] + [i for i in self.queries] + [i for i in self.values] 
        return weights
    def forward(self,value ,key , query, scale = True , mask = False ) :
        

        k = self.k(key)
        q = self.q(query)
        v = self.v(value)
        
        attention , _ = self.att(q , k , v)
        return attention #self.output(attention)
    
class transformerBlock(nn.Module):# A LAYER_NORM AINDA NÃO ME CONVENCEU
    def __init__(self, model_dim , heads , forward_expansion = 4):
        super(transformerBlock , self ).__init__()
        self.attention   = selfAttention(model_dim ,heads= heads)
        self.layerNorm0  = nn.LayerNorm(model_dim )
        self.layerNorm1  = nn.LayerNorm(model_dim )
        self.feedForward = nn.Sequential(nn.Linear(model_dim,model_dim * forward_expansion) , nn.ELU() , nn.Linear(model_dim * forward_expansion ,model_dim))

    def setDevice(self , device ):
        self.attention.to(device)
        self.layerNorm0.to(device)
        self.layerNorm1.to(device)
        self.feedForward[0].to(device)
        self.feedForward[2].to(device)

    def weights(self)->list:
        weights = self.attention.weights()
        weights += [i for i in self.feedForward]
        return weights
    
    def forward(self ,value ,key ,query ,scale = True , mask = False):
        attention = self.attention(value ,key ,query , scale = scale , mask = mask)
        x = self.layerNorm0(attention + query)
        forward = self.feedForward(x)

        return self.layerNorm1(forward + x)
    


class decoderBlock(nn.Module):
    def __init__(self ,model_dim ,heads ,forward_expansion = 4 , device = torch.device("cpu")):
        super(decoderBlock,self).__init__()
        self.attention = selfAttention(model_dim , heads)
        self.norm = nn.LayerNorm(model_dim)
        self.transformerBlock = transformerBlock(model_dim,heads , forward_expansion = forward_expansion)
        self.device = device

    def setDevice(self,device):
        self.attention.to(device)
        self.norm.to(device)
        self.transformerBlock.setDevice(device)
        self.device = device

    def weights(self)->list:
        weights = self.attention.weights()
        
        weights += self.transformerBlock.weights()
        return weights
    def forward(self ,x ,values ,keys ,mask = True ,scale = False) :
        attention = self.attention(x,x,x , mask = mask , scale = scale)
        queries = self.norm(attention + x)
        return self.transformerBlock(values , keys , queries , scale = scale)
class decoder(nn.Module):
    def __init__(self,model_dim ,heads ,num_layers ,#word_Embedding  ,
                 num_Classes , device = torch.device("cpu") , embed_classes = True ,
                 BOS = None ,
                 EOS = None ,
                 forward_expansion = 4 ):
        super(decoder,self).__init__()

        # self.embedding = word_Embedding
        self.device = device
        self.model_dim = model_dim
        self.embed_classes = embed_classes
        self.EOS = Variable(torch.rand(1 , self.model_dim , dtype = float) ,
                             requires_grad = True).float() if EOS == None else EOS #End-Of-Sentence Vector
        self.BOS = Variable(torch.rand(1 , self.model_dim , dtype = float) ,
                             requires_grad = True).float() if BOS == None else BOS #Begin-Of-Sentence Vector
        self.classes = tuple([self.BOS , self.EOS ] + [Variable(torch.rand(1 , self.model_dim , dtype = float) , requires_grad = True).float() for i in torch.arange(num_Classes)])
        
        
        """self.embedding["<BOS>"] = self.BOS#DEPOIS TIRAR ESSE NUMPY
        self.embedding["<EOS>"] = self.EOS#DEPOIS TIRAR ESSE NUMPY"""
        self.layers = nn.ModuleList( decoderBlock(model_dim , heads , forward_expansion = forward_expansion) for _ in torch.arange(num_layers))
        # self.linear_Out = nn.Linear(model_dim , len(self.embedding) )---OLHAR AKI DEPOIS---
        # self.linear_Out = nn.Linear(model_dim , len(self.embedding.vocab) )
        self.linear_Out = nn.Linear(model_dim , num_Classes )
        """if type(EOS) != type(torch.tensor([1])) :
            self.EOS = torch.from_numpy(self.EOS).float()
            self.BOS = -self.EOS"""
        
        self.pos_Encoder = PositionalEncoding(model_dim, device)
    def set_Device(self , device : torch.device):
        self.device = device
    def weights(self)->list:
        weights = [self.linear_Out]
        for i in self.layers :
            weights += i.weights()
        return weights
    
    def forward_fit(self ,Enc_values , Enc_keys , max_lengh  ) :
        sequence = self.BOS
        soft_Out = [] # nn.ModuleList([])
        # if type(sequence) != type(torch.tensor([1])) :
        #     Enc_values = torch.from_numpy(Enc_values).float()
        #     Enc_keys   = torch.from_numpy(Enc_keys).float()
            # sequence   = torch.from_numpy(self.BOS).float()

        while  sequence.shape[0]<= max_lengh  :# Ta errado
            print("Mais um loop de Decoder e sequence.shape[0] = " , sequence.shape[0] )
            buffer = self.pos_Encoder(sequence)
            
            for l in self.layers :
                buffer = l(buffer , Enc_values , Enc_keys)
            buffer = F.softmax(self.linear_Out(buffer[-1]) , dim = 0 )
            out        = torch.argmax(buffer).item()
            # out = heapq.nlargest(1, enumerate(buffer ) , key = lambda x : x[1])[0]
            soft_Out.append(buffer.view(1,-1))
            
            if self.embed_classes :
                sequence = torch.cat((sequence , self.classes[ out ].float().view(1,-1) ),dim = 0 )
                # sequence = torch.cat((sequence , self.embedding.vocabulary[self.embedding.idx2token[out[0]]]),dim = 0 )
                # sequence = torch.cat((sequence , self.classes[ out[0] ].float().view(1,-1)),dim = 0 )
            else :
                if out != 1 :
                    sequence = torch.cat((sequence , self.BOS ),dim = 0 )
                else :
                    sequence = torch.cat((sequence , self.EOS ),dim = 0 )
        return torch.cat(soft_Out ,dim = 0)
        

    def forward(self ,Enc_values , Enc_keys , max_lengh = 100 ) :
        sequence = self.BOS
        idx = [ 0 ] #Ta ERRADO
        while sequence[-1] != self.EOS and sequence.shape[0]< max_lengh  :# Ta errado
            buffer = self.pos_Encoder(sequence)

            for l in self.layers :
                buffer = l(buffer , Enc_values , Enc_keys)
            buffer = F.softmax(self.linear_Out(buffer[-1]) , dim = 0 )
            out        = torch.argmax(buffer).item()
            # out = heapq.nlargest(1, enumerate(buffer ) , key = lambda y : y[1])[0]
            
            idx.append(out)
            # idx.append(out[0])
            #buffer = F.softmax(buffer , dim = 1)
            #buffer = O Vetor com a maior probabilidade , mas qual ??
            
            if self.embed_classes :
                sequence = torch.cat((sequence , self.classes[ out ].float().view(1,-1) ),dim = 0 )
                # sequence = torch.cat((sequence , self.embedding.vocabulary[self.embedding.idx2token[out[0]]]),dim = 0 )
            else :
                if out != 1 :
                    sequence = torch.cat((sequence , self.BOS ),dim = 0 )
                else :
                    sequence = torch.cat((sequence , self.EOS ),dim = 0 )

        sequence = idx #[self.embedding.idx2token[i] for i in idx ]
        return sequence

        if sequence.shape[0] == max_lengh -1 :
            return torch.cat((sequence,self.EOS),dim = 0)
        return sequence

