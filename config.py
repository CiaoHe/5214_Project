import torch
class Config(object):
    embed_size = 300
    hidden_layers = 2
    hidden_size = 64
    output_size = 1
    max_epochs = 100
    hidden_size_linear = 64
    lr = 1e-3
    batch_size = 128
    max_sen_len = None
    dropout_keep = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CharCNNConfig(object):
    num_channels = 256
    linear_size = 256
    output_size = 1
    max_epochs = 100
    lr = 1e-3
    batch_size = 128
    max_sen_len = 150
    dropout_keep = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_size = 300