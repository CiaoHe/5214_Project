import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from utils import *

class BaseNet(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings):
        super(BaseNet, self).__init__()
        self.config = config
        #Embed layer
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def add_loss_op(self, loss_op):
        self.loss_op = loss_op
    
    def add_lr_scheduler(self, scheduler):
        self.scheduler = scheduler
    
    def forward(self, x):
        pass