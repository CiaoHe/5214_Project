import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import conv
from torch.nn.modules.pooling import MaxPool1d
from utils import *
from .Base import BaseNet

class CharCNN(BaseNet):
    def __init__(self, config, vocab_size, embeddings):
        super(CharCNN, self).__init__(config)
        #Embed layer
        self.config = config
        embed_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.embeddings.weight = nn.Parameter(embeddings, requires_grad=False)

        conv1 = nn.Sequential(
            nn.Conv1d(embed_size, self.config.num_channels, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
        ) #(len-6)/3
        conv2 = nn.Sequential(
            nn.Conv1d(self.config.num_channels, self.config.num_channels, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
        ) # (leN-6-18)/9
        conv3 = nn.Sequential(
            nn.Conv1d(self.config.num_channels, self.config.num_channels, kernel_size=3),
            nn.ReLU(),
        ) # (len-6-18-18)/9
        conv4 = nn.Sequential(
            nn.Conv1d(self.config.num_channels, self.config.num_channels, kernel_size=3),
            nn.ReLU(),
        ) # (len - 60) /9
        
        conv_out_size = self.config.num_channels * ((self.config.max_sen_len - 60) // 9)
        linear1 = nn.Sequential(
            nn.Linear(conv_out_size, self.config.linear_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_keep)
        )
        linear2 = nn.Sequential(
            nn.Linear(self.config.linear_size, self.config.linear_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_keep)
        )
        linear3 = nn.Sequential(
            nn.Linear(self.config.linear_size, self.config.output_size),
            nn.Sigmoid()
        )
        
        self.conv_layers = nn.Sequential(conv1, conv2, conv3, conv4)
        self.linear_layers = nn.Sequential(linear1, linear2, linear3)
    
    def forward(self, x):
        embed_sent = self.embeddings(x) # (seq_len, bs, embed_dim)
        embed_sent = embed_sent.permute(1,2,0) #(bs, embed_dim, seq_len)
        conv_out = self.conv_layers(embed_sent) #(bs, embed_dim, 10)
        out = self.linear_layers(conv_out.view(conv_out.shape[0],-1)) 
        return out
