import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from utils import *
from .Base import BaseNet

class TextRNN(BaseNet):
    def __init__(self, config, vocab_size, word_embeddings):
        super().__init__(config)

        #embedding layer
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

        self.lstm = nn.LSTM(
            input_size = self.config.embed_size,
            hidden_size = self.config.hidden_size,
            num_layers = self.config.hidden_layers,
            dropout = self.config.dropout_keep,
            bidirectional = self.config.bidirectional,
        )
        self.dropout = nn.Dropout(self.config.dropout_keep)

        self.fc = nn.Sequential(
            nn.Linear(self.config.hidden_size * self.config.hidden_layers * (1+self.config.bidirectional),
            self.config.output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        embed_sent = self.embeddings(x)
        lstm_out, (h_n,c_n) = self.lstm(embed_sent)
        feat_map = self.dropout(h_n)

        feat_map = torch.cat([
            feat_map[i,:,:] for i in range(feat_map.shape[0])
        ], dim=1)

        out = self.fc(feat_map)
        return out
