import torch
from torch import nn
import math
from torch import nn,optim
import torch.nn.functional as F
from data_manager import*

class BiLSTM_Attention(nn.Module):
    def __init__(self,vocab_size,embeding_dim,hidden_dim,n_layers):
        super(BiLSTM_Attention,self).__init__()
        self.hidden_dim=hidden_dim
        self.n_layers=n_layers
        self.embedding=nn.Embedding(vocab_size,embeding_dim)
        self.rnn=nn.LSTM(embeding_dim,hidden_dim,num_layers=n_layers,bidirectional=True,
                         dropout=0.5)#output:[seq_len,batch,num_directions*hidden_size]
        self.fc=nn.Linear(hidden_dim*2,1)#因为时双层的，所以需要对hidden_dim*2
        self.dropout=nn.Dropout(0.5)

    def attention_net(self,x,query,mask=None):#软性注意力机制（k=v） x->[128,52,128]
        d_k=query.size(-1)#batch_size,seq_len,embedding_dim
        #打分机制 scores:[batch, seq_len, seq_len]
        scores=torch.matmul(query,x.transpose(1,2))/math.sqrt(d_k) #[batch,seq_len,seq_len]
        p_attn=F.softmax(scores,dim=-1) #沿着列方向进行softmax [batch,seq_len,seq_len] ->[128,52,52]
        context=torch.matmul(p_attn,x).sum(1) #将注意力矩阵与输入序列相乘再求和 ->[128,52,52]
        return context,p_attn

    def forward(self,x):
        embedding=self.dropout(self.embedding(x)) #[seq_len,batch,embedding_dim]

        output,(final_hidden_state,final_cell_state)=self.rnn(embedding) #output[seq_len,batch,num_direction*hidden_size]
        output=output.permute(1,0,2) #[batch,seq_len,num_direction*hidden_size]

        query=self.dropout(output) #[batch,seq_len,num_direction*hidden_size]
        attn_output,attention=self.attention_net(output,query) #将LSTM的输出作为key,将dropout后的LSTM输出作为query
        logit=self.fc(attn_output)
        return logit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rnn=BiLSTM_Attention(len(TEXT.vocab),EMBEDDING_DIM,hidden_dim=64,n_layers=2)
pretrained_embedding=TEXT.vocab.vectors
rnn.embedding.weight.data.copy_(pretrained_embedding)
optimizer=optim.Adam(rnn.parameters(),lr=LEARNING_RATE)
criteon=nn.BCEWithLogitsLoss()


