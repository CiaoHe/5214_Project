from model import *
from utils import *
from config import *
import sys
import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

def run_epoch(model, train_iter, val_iter, epoch):
    losses = []
    device = model.config.device

    model.train()
    for batch in train_iter:
        model.zero_grad()
        x = batch.text.to(device)
        y = batch.label.type(torch.float).to(device)
        y_pred = model(x)
        loss = model.loss_op(y_pred, y.view(-1,1))
        loss.backward()
        losses.append(loss.data.cpu().numpy())
        model.optimizer.step()
    
    avg_train_loss = np.mean(losses)
    print('Average Train loss : {:.4f}'.format(avg_train_loss))
    losses = []
    #Eval 
    val_acc, val_f1 = evaluate(model, val_iter)
    print('Val Acc: {:.4f}, F1: {:.4f}'.format(val_acc, val_f1))

if __name__ == '__main__':
    setup_seed(20)
    # config = Config()
    # config = CharCNNConfig()
    config = TextRNNConfig()
    train_file = './data/train.csv'
    test_file = './data/test.csv'
    w2v_file = './data/glove.42B.300d.txt'

    dataset = Dataset(config)
    # dataset = CharCNNDataset(config)
    print('Loading data...')
    dataset.load_data(w2v_file, train_file, test_file)
    # dataset.load_data(train_file, test_file)
    print('Data loaded.')

    # model = RCNN(config, len(dataset.vocab), dataset.word_embeddings)
    # model = CharCNN(config, len(dataset.vocab), dataset.char_embeddings)
    model = TextRNN(config, len(dataset.vocab), dataset.word_embeddings)
    print('===> Current Used Model: %s' % model.__class__.__name__)
    model.to(config.device)
    model.train()
    init_network(model)
    model.add_loss_op(nn.BCELoss())
    model.add_optimizer(optim.Adam(model.parameters(), lr=config.lr))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=model.optimizer, gamma=0.99)
    model.add_lr_scheduler(scheduler)

    #beign train
    for i in range(config.max_epochs):
        print(f"Epoch {i+1}")
        metric = run_epoch(model, dataset.train_iterator, dataset.val_iterator, i)
        model.scheduler.step()
        print('Current lr: {:.4f}'.format(get_lr(model.scheduler)))
    
    train_acc, train_f1 = evaluate(model, dataset.train_iterator)
    val_acc, val_f1 = evaluate(model, dataset.val_iterator)
    print('Final acc on Trainset: {:.4f}, f1: {:.4f}'.format(train_acc, train_f1))
    print('Final acc on Valset: {:.4f}, f1: {:.4f}'.format(val_acc, val_f1))

    #predict(model, dataset.test_iterator)