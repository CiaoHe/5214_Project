import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from utils import *
from .Base import BaseNet

class RCNN(BaseNet):
    def __init__(self, config, vocab_size, word_embeddings):
        super(RCNN, self).__init__(config, vocab_size, word_embeddings)

        #Bi-lstm 
        self.lstm = nn.LSTM(
            input_size = self.config.embed_size,
            hidden_size = self.config.hidden_size,
            num_layers = self.config.hidden_layers,
            dropout = self.config.dropout_keep,
            bidirectional = True,
        )
        self.dropout = nn.Dropout(self.config.dropout_keep)

        self.W = nn.Linear(
            self.config.embed_size + 2*self.config.hidden_size,
            self.config.hidden_size_linear
        )
        
        # Tanh non-linearity
        self.tanh = nn.Tanh()
        
        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.hidden_size_linear,
            self.config.output_size
        )
        
        # Sigmoid non-linearity
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        #x.shape = (seq_len, batch_size)
        embed_sent = self.embeddings(x)
        #embed.shape = (seq_len, batch_size, embed_dim)

        lstm_out, (h_n,c_n) = self.lstm(embed_sent)
        #lstm_out.shape = (seq_len, batch_size, 2*hidden_size)

        input_feats = torch.cat([lstm_out, embed_sent],2).permute(1,0,2)
        #input_feats.shape = (bs,seq_len,2*hidden_size+embed_size)

        linear_output = self.tanh(self.W(input_feats))
        #linear_output.shape = (bs,seq_len,hidden_size_linear)

        #reshape for max-pooling (remains one max seq)
        #linear_output.shape = (bs,hidden_size_linear,seq_len)
        linear_output = linear_output.permute(0,2,1)

        #shape = (bs, hidden_size_linear)
        max_out_features = F.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)

        max_out_features = self.dropout(max_out_features)
        max_out_features = self.fc(max_out_features)
        return torch.sigmoid(max_out_features)

