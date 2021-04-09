import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from utils import *
from .Base import BaseNet

class TextCNN(BaseNet):
    def __init__(self, config, vocab_size, word_embeddings):
        super().__init__(config)
        self.config = config

        #embedding layer
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.config.embed_size, self.config.num_channels,self.config.kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[0]+1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(self.config.embed_size, self.config.num_channels,self.config.kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[1]+1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(self.config.embed_size, self.config.num_channels,self.config.kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[2]+1)
        )

        self.dropout = nn.Dropout(self.config.dropout_keep)

        self.fc = nn.Sequential(
            nn.Linear(self.config.num_channels * len(self.config.kernel_size), self.config.output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        #x.shape (len, batch_size) -> (len, bs, dim) -> (bs, dim, len)
        embed_sent = self.embeddings(x).permute(1,2,0)
        conv1_out = self.conv1(embed_sent).squeeze(2) #(bs,dim,)
        conv2_out = self.conv2(embed_sent).squeeze(2)
        conv3_out = self.conv3(embed_sent).squeeze(2)
        conv_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
        feat_map = self.dropout(conv_out)
        out = self.fc(feat_map)
        return out