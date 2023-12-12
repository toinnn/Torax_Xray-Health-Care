import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
import random as rd
from torch import tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader , Dataset
import copy as cp
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

def diff_Rate(a,b):
    correct = 0
    smallerSize = min(a.shape[1] , b.shape[1])
    for n in range(a.shape[0]):
         #min(a.shape[1] , b.shape[1] )
        for i in range(smallerSize):
            # print("a[i] {} -- b[i] {}".format(a[n][i] , b[n][i]))
            # for j in 
            if a[n][i]==b[n][i] :
                correct += 1 
        # biggerSize = max(len(a) , len(b))
    return 1 - correct/(max(a.shape[1] , b.shape[1] ) * max(a.shape[0] , b.shape[0])) #biggerSize 


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

        self.att = nn.MultiheadAttention(embed_dim= model_dim , num_heads= heads , device=device , batch_first=True)
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
        
        # q = torch.cat([ q.view(1,q.shape[0] , q.shape[1]) for _ in torch.arange(k.shape[0]) ] )
        attention , _ = self.att(q , k , v)
        return attention #self.output(attention)
    
class transformerBlock(nn.Module):# A LAYER_NORM AINDA NÃO ME CONVENCEU
    def __init__(self, model_dim , heads , forward_expansion = 4 , device = torch.device("cpu")):
        super(transformerBlock , self ).__init__()
        self.attention   = selfAttention(model_dim ,heads= heads)
        self.layerNorm0  = nn.LayerNorm(model_dim )
        self.layerNorm1  = nn.LayerNorm(model_dim )
        self.feedForward = nn.Sequential(nn.Linear(model_dim,model_dim * forward_expansion) , nn.ELU() , nn.Linear(model_dim * forward_expansion ,model_dim))
        self.device = device

    def setDevice(self , device ):
        self.attention.setDevice(device)
        self.layerNorm0.to(device)
        self.layerNorm1.to(device)
        self.feedForward[0].to(device)
        self.feedForward[2].to(device)
        self.device = device

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
                 BOS = None  ,
                 EOS = None  ,
                 forward_expansion = 4 , classification = True):
        super(decoder,self).__init__()

        self.device = device
        self.model_dim = model_dim
        self.type_net = None

        self.linear_Out = None
        if classification :
            if num_Classes == 1 :
                self.type_net = "Bin_classification"
            else :
                self.type_net = "classification"
            self.linear_Out = nn.Linear(model_dim , num_Classes )
        else :
            self.type_net = "regression"
            # self.linear_Out = nn.Linear(model_dim , num_Classes )
            self.linear_Out = nn.Linear(model_dim , 1 )

        


        ## self.type_net == "classification" :
        self.embed_classes = embed_classes
        self.EOS = Variable(torch.rand(1 , self.model_dim , dtype = float) ,
                             requires_grad = True).float() if EOS == None else EOS #End-Of-Sentence Vector
        self.BOS = Variable(torch.rand(1 , self.model_dim , dtype = float) ,
                             requires_grad = True).float() if BOS == None else BOS #Begin-Of-Sentence Vector
        self.classes = [ self.EOS ] + [Variable(torch.rand(1 , self.model_dim , dtype = float) , requires_grad = True).float() for i in torch.arange(num_Classes)]
        
        
        self.layers = nn.ModuleList( decoderBlock(model_dim , heads , forward_expansion = forward_expansion) for _ in torch.arange(num_layers))
        # print(f"decoder layers : {self.layers} ")
        
        
        self.pos_Encoder = PositionalEncoding(model_dim, device)

    def setDevice(self , device : torch.device):
        self.device = device
        self.layers = self.layers.to(device) #nn.ModuleList( i.setDevice(device) for i in self.layers )
        self.linear_Out = self.linear_Out.to(device)
        self.BOS = self.BOS.to(device)
        self.EOS = self.EOS.to(device)
        self.classes = [  i.to(device) for i in self.classes ]
        self.pos_Encoder.setDevice(device)

    def weights(self)->list:
        weights = [self.linear_Out]
        for i in self.layers :
            weights += i.weights()
        return weights
    
   
    def forward(self ,Enc_values , Enc_keys , max_lengh = 100  , force_max_lengh = False) : #out shape = batch , seq-len
        
        if self.type_net    == "classification" :
            sequence  = self.__forward_classification(Enc_values , Enc_keys , max_lengh   , force_max_lengh)
            return sequence[: , -1]
        elif self.type_net  == "Bin_classification" :
            
            buffer = self.__Bin_classification( Enc_values , Enc_keys)
            return torch.round( buffer )
            
        elif self.type_net  == "regression" :
            pass
        
    def forward_fit(self ,Enc_values , Enc_keys , max_lengh = None ) : #out shape = batch , seq-len , número de classes
        
        
        if self.type_net     == "classification" :

            if max_lengh == None or type(max_lengh) != type(1) :
                raise ValueError("max_lengh deve ser preenchido com um inteiro igual ao tamanho da sequência target durante durante a fase de treinamento de uma rede Seq2Seq")
            return self.__forward_fit_classification(Enc_values , Enc_keys , max_lengh )
        
        elif self.type_net   == "Bin_classification" :

            return self.__Bin_classification( Enc_values , Enc_keys)
         
        elif self.type_net   == "regression" :
            pass



    def __forward_fit_classification(self , Enc_values , Enc_keys , max_lengh ) :

        sequence = torch.cat( [self.BOS.view(1,1,-1) for _ in torch.arange(Enc_values.shape[0])]  , dim = 0 )
        soft_Out = [] # nn.ModuleList([])
        # if type(sequence) != type(torch.tensor([1])) :
        #     Enc_values = torch.from_numpy(Enc_values).float()
        #     Enc_keys   = torch.from_numpy(Enc_keys).float()
            # sequence   = torch.from_numpy(self.BOS).float()

        while  sequence.shape[1]<= max_lengh  : #Comprimento da Sequencia
            # print("Mais um loop de Decoder e sequence.shape = " , sequence.shape )
            buffer = self.pos_Encoder(sequence)
            # buffer = torch.cat([ q.view(1,buffer.shape[0] , buffer.shape[1]) for _ in torch.arange(Enc_keys.shape[0]) ] ,
            #               dim = 0 )
            
            for l in self.layers :
                # print(f"\nbuffer : {buffer.shape}, \nEnc_values : {Enc_values.shape}, \nEnc_keys : {Enc_keys.shape}")
                buffer = l(buffer , Enc_values , Enc_keys)
            
            # buffer = F.softmax(self.linear_Out(buffer[-1]) , dim = 0 )
            buffer = torch.cat( list( self.linear_Out(buffer[i][-1]).view(1 ,1,-1) for i in torch.arange(buffer.shape[0]) ) ,
                               dim = 0)
            buffer = F.softmax(buffer , dim = 2)

            # list( torch.argmax(out[i][-1] ) for i in torch.arange(out.shape[0]) )
            out    = torch.argmax(buffer , dim = 2)
            # out        = torch.argmax(buffer).item()
            # out = heapq.nlargest(1, enumerate(buffer ) , key = lambda x : x[1])[0]
            soft_Out.append(buffer)

            # print(f"out.shape : {out.shape}")
            
            if self.embed_classes :
                # aux = torch.cat([ self.classes[ i[-1][0] ].float().view(1, 1,-1) for i in out ] , dim = 0)
                aux = torch.cat([ self.classes[ i[-1].item() ].float().view(1, 1,-1) for i in out ] , dim = 0)
                sequence = torch.cat((sequence , aux) , dim = 1 )
                # sequence = torch.cat((sequence , self.embedding.vocabulary[self.embedding.idx2token[out[0]]]),dim = 0 )
                # sequence = torch.cat((sequence , self.classes[ out[0] ].float().view(1,-1)),dim = 0 )
            else :
                aux = torch.zeros(out.shape[0] , 1 , self.model_dim , device = self.device )
                for i in torch.arange(out.shape[0]) :
                    if out[i][-1].item() != 1 :
                        aux[i][-1] = self.BOS.view(-1)
                        # sequence = torch.cat((sequence , self.BOS ),dim = 0 )
                    else :
                        aux[i][-1] = self.EOS.view(-1)
                        # sequence = torch.cat((sequence , self.EOS ),dim = 0 )
                sequence = torch.cat((sequence , aux) , dim = 1)
        return torch.cat(soft_Out ,dim = 1)

    def __Bin_classification(self , Enc_values , Enc_keys):    
        sequence = torch.cat( [self.BOS.view(1,1,-1) for _ in torch.arange(Enc_values.shape[0])]  , dim = 0 )
        # soft_Out = []

        buffer = self.pos_Encoder(sequence)
        # buffer = torch.cat([ q.view(1,buffer.shape[0] , buffer.shape[1]) for _ in torch.arange(Enc_keys.shape[0]) ] ,
        #               dim = 0 )
        
        for l in self.layers :
            # print(f"\nbuffer : {buffer.shape}, \nEnc_values : {Enc_values.shape}, \nEnc_keys : {Enc_keys.shape}")
            buffer = l(buffer , Enc_values , Enc_keys)

        buffer = torch.cat( list( self.linear_Out(buffer[i][-1]).view(1 ,1,-1) for i in torch.arange(buffer.shape[0]) ) ,
                               dim = 0)
        
        buffer = F.sigmoid( buffer )
        # class_out = torch.round( buffer )

        return buffer

    def __forward_classification(self , Enc_values , Enc_keys , max_lengh   , force_max_lengh ) :
        
        sequence = torch.cat( [self.BOS.view(1,1,-1) for _ in torch.arange(Enc_values.shape[0])]  , dim = 0 )
        idx = [ 0 ] 
        all_seq_EOS = False
        while ( all_seq_EOS != True and sequence.shape[1]< max_lengh )  or force_max_lengh:# Ta errado
            buffer = self.pos_Encoder(sequence)
            

            for l in self.layers :
                buffer = l(buffer , Enc_values , Enc_keys)

            # buffer = F.softmax(self.linear_Out(buffer[-1]) , dim = 0 )
            
            # out        = torch.argmax(buffer).item()
            # out = heapq.nlargest(1, enumerate(buffer ) , key = lambda y : y[1])[0]



            buffer = torch.cat( list( self.linear_Out(buffer[i][-1]).view(1 ,1,-1) for i in torch.arange(buffer.shape[0]) ) ,
                               dim = 0) # Como o original é feito assim ele terá shape : Batch , 1 , classes
            buffer = F.softmax(buffer , dim = 2) 
            out    = torch.argmax(buffer , dim = 2) #Reduz de 3 dimensões para duas , onde cada linha é um batch do original
            #                                       e cada coluna é uma linha do original 

            idx.append(out)
            # idx.append(out[0])
            #buffer = F.softmax(buffer , dim = 1)
            #buffer = O Vetor com a maior probabilidade , mas qual ??
            
            if self.embed_classes :
                aux = torch.cat([ self.classes[ i[0] ].float().view(1, 1,-1) for i in out ] , dim = 0)
                sequence = torch.cat( (sequence , aux) , dim = 1 )
                # sequence = torch.cat((sequence , self.embedding.vocabulary[self.embedding.idx2token[out[0]]]),dim = 0 )
                # sequence = torch.cat((sequence , self.classes[ out[0] ].float().view(1,-1)),dim = 0 )
            else :
                aux = torch.zeros(out.shape[0] , 1 , self.model_dim , device = self.device )
                for i in torch.arange(out.shape[0]) :
                    if out[i][-1].item() != 0 :
                        aux[i][-1] = self.BOS.view(-1)
                        # sequence = torch.cat((sequence , self.BOS ),dim = 0 )
                    else :
                        aux[i][-1] = self.EOS.view(-1)
                        # sequence = torch.cat((sequence , self.EOS ),dim = 0 )
                sequence = torch.cat((sequence , aux) , dim = 1)
            
            all_seq_EOS = True
            for last_out in idx[-1] :
                if last_out[0].item() != 0 :
                    all_seq_EOS = False
                    break

            if sequence.shape[1] == max_lengh :
                force_max_lengh = False

        # print(f"idx = {idx}")
        idx[0] = torch.zeros(idx[-1].shape[0] , idx[-1].shape[1] , device = self.device )
        sequence = torch.cat(idx , dim = 1 ) #[self.embedding.idx2token[i] for i in idx ]

        return sequence 

     
    def fit(self , input_Batch :list , target_Batch : list, n , maxErro , maxAge = 1 ,mini_batch_size = 1  ,
            lossFunction = nn.CrossEntropyLoss() ,lossGraphPath = None , test_Input_Batch = None,
            test_Target_Batch = None , out_max_Len  = 150 , transform = None) :

        optimizer = torch.optim.Adam(self.parameters(), n )
        lossValue = float("inf")
        Age = 0
        lossList = []
        bestLossValue = float("inf")
        # input_Batch = [i.view(1 , i.shape[0] , i.shape[1] ) for i in input_Batch ]    

        if test_Input_Batch != None and test_Target_Batch != None :
            lossTestList = []
        best_params = cp.deepcopy((self.layers , self.linear_Out , self.classes ))
    
        while lossValue > maxErro and Age < maxAge :
            lossValue = 0
            ctd = 0
            print("Age atual {}".format(Age))
            
            
            best_params , lossValue , lossTestList = self.train_Step(input_Batch , target_Batch , optimizer  ,
             lossFunction ,bestLossValue ,ctd ,lossValue , test_Input_Batch , test_Target_Batch , out_max_Len ,
             best_params,lossTestList , transform )
            
            """for x,y in zip(input_Batch , target_Batch ) :
                if type(y) != type(torch.tensor([1])) :
                    x = torch.from_numpy(x).float()
                    y = torch.from_numpy(y).float()
                div = len(y)
                                
                out = self.forward_fit(x , out_max_Len = y.shape[0] ,target = y.to(self.device) )

                print("Age atual {} , ctd atual {}\nout.shape = {} , y.shape = {}".format(Age ,ctd ,out.shape , y.shape))
                loss = lossFunction(out , y.to(self.device))/div
                lossValue += loss.item()
                print("Pré backward")
                loss.backward()
                print("Pós backward")
                optimizer.step()
                optimizer.zero_grad()
                ctd += 1
            if test_Input_Batch != None and test_Target_Batch != None  :
                diff = 0
                div = min( len(test_Input_Batch) , len(test_Target_Batch) )
                for x,y in zip( test_Input_Batch , test_Target_Batch ) :
                    if type(y) != type(torch.tensor([1])) :
                        x = torch.from_numpy(x).float()
                        y = torch.from_numpy(y).float()

                    _ , out = self.forward(x.to(self.device) , out_max_Len = out_max_Len )
                    diff += diff_Rate(out , y.to(self.device) )
                    
                lossTestList += [diff/div]
                if  lossTestList[-1] < bestLossValue :
                    print("Novo melhor")
                    best_Encoder  =  cp.deepcopy(self.encoder)
                    best_Decoder  =  cp.deepcopy(self.decoder)
                    bestLossValue =  lossTestList[-1]
                    print("Saiu do Melhor")"""

            Age += 1
            lossValue = lossValue/len(target_Batch)
            lossList.append(lossValue)
        
        if test_Input_Batch != None and test_Target_Batch != None  :
            print("O melhor resultado de teste foi " , bestLossValue )
            # self.encoder = cp.deepcopy(best_Encoder)
            self.layers     = best_params[0] 
            self.linear_Out = best_params[1] 
            self.classes    = best_params[2]
            self.BOS = self.classes[0]
            self.EOS = self.classes[1]
            # self.layers best_params = (self.layers , self.linear_Out , self.classes )
        
        # self.__saveLossGraph(lossGraphPath  , Age  , lossList  , bestLossValue , lossTestList)
        """    trainLossPlot = plt.subplot(2,1,1)
            trainLossPlot.plot(range(1 , Age + 1) , lossList)
            plt.ylabel("Loss in Train" , fontsize = 14 )
            plt.xlabel("Ages" , fontsize = 14)

            testLossPlot = plt.subplot(2,1,2)
            testLossPlot.plot(range(1 , Age + 1) , lossTestList )
            plt.ylabel("Test Percent Loss" , fontsize = 14 )
            plt.xlabel("Ages" , fontsize = 14)
        else :
            trainLossPlot = plt.subplot(1 , 1 , 1)
            trainLossPlot.plot(range(1 , Age + 1) , lossList)
            plt.ylabel("Loss in Train" , fontsize = 14 )
            plt.xlabel("Ages" , fontsize = 14)

        if lossGraphPath != None and test_Input_Batch != None and test_Target_Batch != None :
            plt.savefig(f"{lossGraphPath}_BiLSTM_ATTENTON_LossInTrain_Plot.png" )
            plt.savefig(f"{lossGraphPath}_BiLSTM_ATTENTON_LossInTrain_Plot.pdf" )
        else :
            plt.savefig("BiLSTM_ATTENTON_LossInTrain_Plot.png")
            plt.savefig("BiLSTM_ATTENTON_LossInTrain_Plot.pdf")"""
        plt.show()
    
    def train_Step(self ,input_Batch :list , target_Batch : list , optimizer , lossFunction ,bestLossValue : float ,
                   
        ctd : int , lossValue : int , test_Input_Batch= None , test_Target_Batch = None ,  out_max_Len = 150 ,
        best_params = None ,  lossTestList = [] , transform = None ) :
        
        for x,y in zip(input_Batch , target_Batch ) :
            if transform != None :
                x , y = transform(x) , transform(y)
            if type(y) != type(torch.tensor([1])) :
                x = torch.from_numpy(x).float()
                y = torch.from_numpy(y).float()
            div = sum(( i.shape[0] for i in y))#len(y)
                            
            out = self.forward_fit(x , x , out_max_Len = y.shape[1] ) # ,target = y.to(self.device) )(TALVEZ EU RE-EMPLEMENTE A TÉCNICA QUE USA O ARGUMENTO "target")
            out = torch.cat((i[-1].view(1,-1) for i in out ) , dim = 0 )

            print(" ctd atual {}  samples processados {}\nout.shape = {} , y.shape = {}".format(ctd ,ctd*x.shape[0] ,out.shape , y.shape))
            loss = lossFunction(out , y.to(self.device))/div
            lossValue += loss.item()
            print("Pré backward")
            loss.backward()
            print("Pós backward")
            optimizer.step()
            optimizer.zero_grad()
            ctd += 1
        if test_Input_Batch != None and test_Target_Batch != None and best_Decoder != None  :
            diff = 0
            div = min( len(test_Input_Batch) , len(test_Target_Batch) )
            for x,y in zip( test_Input_Batch , test_Target_Batch ) :
                if transform != None :
                    x , y = transform(x) , transform(y)
                if type(y) != type(torch.tensor([1])) :
                    x = torch.from_numpy(x).float()
                    y = torch.from_numpy(y).float()

                out = self.forward(x.to(self.device) , x.to(self.device) , out_max_Len = out_max_Len )
                diff += diff_Rate(out , y.to(self.device) )
                
            lossTestList += [diff/div]
            print(f"lossTestList : {lossTestList}")
            if  lossTestList[-1] < bestLossValue :
                print("Novo melhor")
                # best_Encoder  =  cp.deepcopy(self.encoder) 

                """best_Decoder_layers  =  cp.deepcopy(layers)
                best_linear_Out = cp.deepcopy(linear_Out)
                best_classes = cp.deepcopy(classes)
                best_BOS = best_classes[0]
                best_EOS = best_classes[1]"""

                best_params = cp.deepcopy(( self.layers , self.linear_Out , self.classes))


                bestLossValue =  lossTestList[-1]
                print("Saiu do Melhor")
        
        if test_Input_Batch != None and test_Target_Batch != None  :
            return best_params , lossValue , lossTestList
        else :
            return _ , _ , lossValue , _

        """def __saveLossGraph(self , path2Save :str  , Age : int , lossList : list , bestLossValue : float = None ,
        lossTestList : list = None ):
        if test_Input_Batch != None and test_Target_Batch != None  :
            print("O melhor resultado de teste foi " , bestLossValue )
            self.encoder = cp.deepcopy(best_Encoder)
            self.decoder = cp.deepcopy(best_Decoder)
        
            trainLossPlot = plt.subplot(2,1,1)
            trainLossPlot.plot(range(1 , Age + 1) , lossList)
            plt.ylabel("Loss in Train" , fontsize = 14 )
            plt.xlabel("Ages" , fontsize = 14)

            testLossPlot = plt.subplot(2,1,2)
            testLossPlot.plot(range(1 , Age + 1) , lossTestList )
            plt.ylabel("Test Percent Loss" , fontsize = 14 )
            plt.xlabel("Ages" , fontsize = 14)
        else :
            trainLossPlot = plt.subplot(1 , 1 , 1)
            trainLossPlot.plot(range(1 , Age + 1) , lossList)
            plt.ylabel("Loss in Train" , fontsize = 14 )
            plt.xlabel("Ages" , fontsize = 14)

        if path2Save != None and test_Input_Batch != None and test_Target_Batch != None :
            plt.savefig(f"{path2Save}_BiLSTM_ATTENTON_LossInTrain_Plot.png" )
            plt.savefig(f"{path2Save}_BiLSTM_ATTENTON_LossInTrain_Plot.pdf" )
        else :
            plt.savefig("BiLSTM_ATTENTON_LossInTrain_Plot.png")
            plt.savefig("BiLSTM_ATTENTON_LossInTrain_Plot.pdf")"""


class Trainer():

    def __init__(self , model :decoder , device = torch.device("cpu") ) -> None:
        self.model  = model
        self.device = device 
        self.model.setDevice(device)
        
        
    def deepcopy(self):
        # print(f"O EOS é leaf ? : {self.model.decoder.is_leaf}")
        EOS = cp.deepcopy(self.model.decoder.EOS.detach())
        BOS = cp.deepcopy(self.model.decoder.BOS.detach())
        classes = cp.deepcopy([i.detach() for i in self.model.decoder.classes ] )
        layers  = cp.deepcopy( self.model.decoder.layers )#nn.ModuleList( [i.detach() for i in self.model.decoder.layers  ] ) )

        linear_Out  = cp.deepcopy(self.model.decoder.linear_Out)

        return (EOS , BOS , classes , layers  , linear_Out) 

    def loadcopy(self , params): 
        
        print(params)
        self.model.EOS = params[0].requires_grad()
        self.model.BOS = params[1].requires_grad()
        self.model.classes = [ i.requires_grad() for i in params[2] ]
        self.model.layers  = params[3]
        self.model.linear_Out  = params[4]

        self.model.setDevice(self.device)

    def fit(self , dataloader : DataLoader ,n , maxErro , maxAge = 1 ,mini_batch_size = 1  , #input_Batch :list , target_Batch : list, n , maxErro , maxAge = 1 ,mini_batch_size = 1  ,
            lossFunction = nn.CrossEntropyLoss ,lossGraphPath = None , test_dataloader = None ,#Input_Batch = None,
            out_max_Len  = 150  , transform = None ,
            model_class  = None , model_args :dict = {} , class_weight = None ) : #test_Target_Batch = None , out_max_Len  = 150 , transform = None) :

        if class_weight != None :
            lossFunction = lossFunction(weight = class_weight )
        else :
            lossFunction = lossFunction()
        optimizer = torch.optim.Adam(self.model.parameters(), n )
        lossValue = float("inf")
        Age = 0
        lossList = []
        bestLossTestValue = float("inf")
        # input_Batch = [i.view(1 , i.shape[0] , i.shape[1] ) for i in input_Batch ]    

        if test_dataloader != None : #test_Input_Batch != None and test_Target_Batch != None :
            lossTestList = []
            
        best_params = self.deepcopy()#
        # best_params = cp.deepcopy(self.model.detach() )

        while lossValue > maxErro and Age < maxAge :
            lossValue = 0
            ctd = 0
            print("Age atual {}".format(Age))
            
            #                                                           dataloader
            best_params , lossValue , lossTestList , bestLossTestValue = self.train_Step(dataloader , optimizer  ,
             lossFunction ,bestLossTestValue ,ctd ,lossValue , test_dataloader , out_max_Len ,
             best_params,lossTestList , transform , test_inside_age = True , test_interval = 2 )
            
            """for x,y in zip(input_Batch , target_Batch ) :
                if type(y) != type(torch.tensor([1])) :
                    x = torch.from_numpy(x).float()
                    y = torch.from_numpy(y).float()
                div = len(y)
                                
                out = self.forward_fit(x , out_max_Len = y.shape[0] ,target = y.to(self.device) )

                print("Age atual {} , ctd atual {}\nout.shape = {} , y.shape = {}".format(Age ,ctd ,out.shape , y.shape))
                loss = lossFunction(out , y.to(self.device))/div
                lossValue += loss.item()
                print("Pré backward")
                loss.backward()
                print("Pós backward")
                optimizer.step()
                optimizer.zero_grad()
                ctd += 1
            if test_Input_Batch != None and test_Target_Batch != None  :
                diff = 0
                div = min( len(test_Input_Batch) , len(test_Target_Batch) )
                for x,y in zip( test_Input_Batch , test_Target_Batch ) :
                    if type(y) != type(torch.tensor([1])) :
                        x = torch.from_numpy(x).float()
                        y = torch.from_numpy(y).float()

                    _ , out = self.forward(x.to(self.device) , out_max_Len = out_max_Len )
                    diff += diff_Rate(out , y.to(self.device) )
                    
                lossTestList += [diff/div]
                if  lossTestList[-1] < bestLossValue :
                    print("Novo melhor")
                    best_Encoder  =  cp.deepcopy(self.encoder)
                    best_Decoder  =  cp.deepcopy(self.decoder)
                    bestLossValue =  lossTestList[-1]
                    print("Saiu do Melhor")"""

            Age += 1
            lossValue = lossValue/( len(dataloader.dataset) / dataloader.batch_size )
            lossList.append(lossValue)
        
        if test_dataloader != None : #and test_Target_Batch != None  :
            print("O melhor resultado de teste foi " , bestLossTestValue )
            # self.encoder = cp.deepcopy(best_Encoder)
            """self.layers     = best_params[0] 
            self.linear_Out = best_params[1] 
            self.classes    = best_params[2]
            self.BOS = self.classes[0]
            self.EOS = self.classes[1]"""
            # self.model = best_params.requires_grad()
            
            #Em testes pode ser que mude :
            # self.loadcopy(best_params)
            self.model = self.load_from_Path("best_model_in_test.pickle" , model_class , model_args )
            
            # self.layers best_params = (self.layers , self.linear_Out , self.classes )
        
        # self.__saveLossGraph(lossGraphPath  , Age  , lossList  , bestLossValue , lossTestList)
        """    trainLossPlot = plt.subplot(2,1,1)
            trainLossPlot.plot(range(1 , Age + 1) , lossList)
            plt.ylabel("Loss in Train" , fontsize = 14 )
            plt.xlabel("Ages" , fontsize = 14)

            testLossPlot = plt.subplot(2,1,2)
            testLossPlot.plot(range(1 , Age + 1) , lossTestList )
            plt.ylabel("Test Percent Loss" , fontsize = 14 )
            plt.xlabel("Ages" , fontsize = 14)
        else :
            trainLossPlot = plt.subplot(1 , 1 , 1)
            trainLossPlot.plot(range(1 , Age + 1) , lossList)
            plt.ylabel("Loss in Train" , fontsize = 14 )
            plt.xlabel("Ages" , fontsize = 14)

        if lossGraphPath != None and test_Input_Batch != None and test_Target_Batch != None :
            plt.savefig(f"{lossGraphPath}_BiLSTM_ATTENTON_LossInTrain_Plot.png" )
            plt.savefig(f"{lossGraphPath}_BiLSTM_ATTENTON_LossInTrain_Plot.pdf" )
        else :
            plt.savefig("BiLSTM_ATTENTON_LossInTrain_Plot.png")
            plt.savefig("BiLSTM_ATTENTON_LossInTrain_Plot.pdf")"""
        # plt.show()

        return self.model , min(lossTestList)
    
    def __save_In_Path(  self , model , path ) :
        torch.save( model.state_dict(), path )
        # with open( path , 'wb') as file:
        #     pickle.dump( model, file)
        print(f"Objeto salvo em { path }")


    def load_from_Path(self , path , model_class , model_args : dict ) :
        instance = model_class(**model_args)
        instance.load_state_dict(torch.load(path))
        
        # with open(path, 'rb') as file:
        #     obj = pickle.load(file)
        # print(f"Objeto carregado de {path}")
        # return obj
        return instance
    

    def __teste(self , test_dataloader ,best_params , out_max_Len , lossTestList , bestLossValue , transform = None) :
        diff = 0
        div  = len(test_dataloader.dataset )#min( len(test_Input_Batch) , len(test_Target_Batch) )
        for x,y in test_dataloader : #zip( test_Input_Batch , test_Target_Batch ) :
            if transform != None :
                x , y = transform(x) , transform(y)
            if type(y) != type(torch.tensor([1])) :
                x = torch.from_numpy(x).float()
                y = torch.from_numpy(y).float()
            
            x = torch.cat( [i for i in x ] , dim = 0 ).to(self.model.device)
            y = torch.cat( [i for i in y ] , dim = 1 ).to(self.model.device)

            out = self.model.forward(x.to(self.device) , max_lengh = out_max_Len )
            diff += diff_Rate(out , y.to(self.device).transpose(0,1) )
            
        lossTestList += [diff/div]
        print(f"lossTestList : {lossTestList}\nbestLossValue = {bestLossValue}")
        if  lossTestList[-1] < bestLossValue :
            print("Novo melhor")
            # best_Encoder  =  cp.deepcopy(self.encoder) 

            """best_Decoder_layers  =  cp.deepcopy(layers)
            best_linear_Out = cp.deepcopy(linear_Out)
            best_classes = cp.deepcopy(classes)
            best_BOS = best_classes[0]
            best_EOS = best_classes[1]"""

            best_params = self.deepcopy() #cp.deepcopy(self.model)
            # best_params = cp.deepcopy(self.model.detach())

            path = "best_model_in_test.pickle"
            bestLossValue =  lossTestList[-1]
            self.__save_In_Path( self.model , path )
            # print(f"Novo bestLossValue = {bestLossValue}")
            print("Saiu do Melhor")

        return best_params , bestLossValue , lossTestList  


    def train_Step(self , dataloader , optimizer , lossFunction , bestLossValue : float , #input_Batch :list , target_Batch : list , optimizer , lossFunction ,bestLossValue : float ,
        ctd : int , lossValue : int ,test_dataloader = None ,# test_Input_Batch= None , test_Target_Batch = None ,  out_max_Len = 150 ,
        out_max_Len = 150 , best_params = None ,  lossTestList = [] , transform = None ,
        test_inside_age = False , test_interval : int = 100  ) :
        
        for x,y in dataloader :
            
            if transform != None :
                x , y = transform(x) , transform(y)
            if type(y) != type(torch.tensor([1])) :
                x = torch.from_numpy(x).float()
                y = torch.from_numpy(y).int()
            
            x = torch.cat( [i for i in x ] , dim = 0 ).to(self.model.device)
            y = torch.cat( [i for i in y ] , dim = 1 ).to(self.model.device)

            div = y.shape[1]*y.shape[0]#sum(( i.shape[0] for i in y))#len(y)
            
            # print(f"x : {x.shape}\ny : {y.shape}")
            # raise
            # print(f"y.shape[0] = {y.shape[0]}")
            out = self.model.forward_fit(x ,  max_lengh = y.shape[0] ) # ,target = y.to(self.device) )(TALVEZ EU RE-EMPLEMENTE A TÉCNICA QUE USA O ARGUMENTO "target")
            out = out.transpose(1 , 2)
            y   = y.transpose(  0 , 1).type(torch.LongTensor).to(self.device)
            # print(f"out = {out.shape}")
            # raise
            # out = torch.cat((i[-1].view(1,-1) for i in out ) , dim = 0 )

            print(" ctd atual {}  samples processados {}\nout.shape = {} , y.shape = {}".format(ctd , ctd*x.shape[0] ,out.shape , y.shape))
            loss = lossFunction(out , y )/div
            lossValue += loss.item()
            print(f"loss : {loss.item()}")
            print("Pré backward")
            loss.backward()
            print("Pós backward")
            optimizer.step()
            optimizer.zero_grad()
            ctd += 1

            if test_inside_age and (ctd % test_interval == 0) and test_dataloader != None and best_params != None : 
                # A CADA 100 ITERAÇÕES DE MINIBATCH É INICIADA UMA ROTINA DE TESTE
                # print("Entrou no teste")
                best_params , bestLossValue , lossTestList  = self.__teste(test_dataloader ,best_params , out_max_Len , lossTestList , bestLossValue , transform )
            
        if test_dataloader != None and best_params != None : #test_Input_Batch != None and test_Target_Batch != None    :
            
            best_params , bestLossValue , lossTestList  = self.__teste(test_dataloader ,best_params , out_max_Len , lossTestList , bestLossValue , transform )
            
        
        if test_dataloader != None  :
            return best_params , lossValue , lossTestList , bestLossValue
        else :
            return _ , _ , lossValue , _

        """def __saveLossGraph(self , path2Save :str  , Age : int , lossList : list , bestLossValue : float = None ,
        lossTestList : list = None ):
        if test_Input_Batch != None and test_Target_Batch != None  :
            print("O melhor resultado de teste foi " , bestLossValue )
            self.encoder = cp.deepcopy(best_Encoder)
            self.decoder = cp.deepcopy(best_Decoder)
        
            trainLossPlot = plt.subplot(2,1,1)
            trainLossPlot.plot(range(1 , Age + 1) , lossList)
            plt.ylabel("Loss in Train" , fontsize = 14 )
            plt.xlabel("Ages" , fontsize = 14)

            testLossPlot = plt.subplot(2,1,2)
            testLossPlot.plot(range(1 , Age + 1) , lossTestList )
            plt.ylabel("Test Percent Loss" , fontsize = 14 )
            plt.xlabel("Ages" , fontsize = 14)
        else :
            trainLossPlot = plt.subplot(1 , 1 , 1)
            trainLossPlot.plot(range(1 , Age + 1) , lossList)
            plt.ylabel("Loss in Train" , fontsize = 14 )
            plt.xlabel("Ages" , fontsize = 14)

        if path2Save != None and test_Input_Batch != None and test_Target_Batch != None :
            plt.savefig(f"{path2Save}_BiLSTM_ATTENTON_LossInTrain_Plot.png" )
            plt.savefig(f"{path2Save}_BiLSTM_ATTENTON_LossInTrain_Plot.pdf" )
        else :
            plt.savefig("BiLSTM_ATTENTON_LossInTrain_Plot.png")
            plt.savefig("BiLSTM_ATTENTON_LossInTrain_Plot.pdf")"""

