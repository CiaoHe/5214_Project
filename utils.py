from pandas._config import config
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.nn.functional import embedding
from torchtext.legacy import data
from torchtext.legacy.data.iterator import BucketIterator
from torchtext.vocab import Vectors
import spacy
import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score, f1_score
import re
from text_preprocess import preprocess

def clean_text(text):
    #2. remove unkonwn characrters
    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    #1. remove http links
    url = re.compile(r'https?://\S+|www\.\S+')
    text = url.sub(r'',text)
    
    #3,4. remove #,@ and othet symbols
    text = text.replace('#',' ')
    text = text.replace('@',' ')
    symbols = re.compile(r'[^A-Za-z0-9 ]')
    text = symbols.sub(r'',text)
    
    #5. lowercase
    text = text.lower()
    
    return text

def get_embedding_matrix(vocab_chars):
    # one hot embedding plus all-zero vector
    vocab_size = len(vocab_chars)
    one_hot_matrix = np.eye(vocab_size, vocab_size)
    return one_hot_matrix

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.val_iterator = None
        self.test_iterator = None
        self.vocab = {}
        self.word_embeddings = {}
    
    def parse_label(self, label):
        return int(label)
    
    def get_pandas_df(self, PATH, mode='train'):
        df = pd.read_csv(PATH)
        data_text = df.text.tolist()
        if mode in ['train','val']:
            labels = df.target.tolist()
            data_label = list(map(self.parse_label, labels))
            new_df = pd.DataFrame({"text":data_text, "label":data_label})
        else:
            new_df = pd.DataFrame({"text":data_text,})
        return new_df

    def load_data(self, w2v_file, train_file, test_file, val_file=None):
        NLP = spacy.load("en_core_web_sm")
        text_preprocessor = preprocess
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(text_preprocessor(sent))]

        TEXT = data.Field(sequential = True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        train_datafields = [('text',TEXT),('label',LABEL)]
        test_datafields = [('text', TEXT)]

        #load train and testt data into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file, mode='train')
        train_examples = [data.Example.fromlist(i, train_datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, train_datafields)

        test_df = self.get_pandas_df(test_file, mode='test')
        test_examples = [data.Example.fromlist(i, test_datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, test_datafields)

        #if validation file exists, then load in the train way; otherwise spilt train_data
        if val_file:
            val_df = self.get_pandas_df(val_file, mode='val')
            val_examples = [data.Example.fromlist(i, train_datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, train_datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)
        
        TEXT.build_vocab(train_data, vectors = Vectors(w2v_file))
        self.word_embeddings = TEXT.vocab.vectors
        self.vocab = TEXT.vocab 

        #get train/val[if]/test dataiterator
        self.train_iterator = data.BucketIterator(
            train_data,
            batch_size = self.config.batch_size,
            sort_key = lambda x: len(x.text),
            repeat = False,
            shuffle = True,
        )
        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size = self.config.batch_size,
            sort_key = lambda x: len(x.text),
            repeat = False,
            shuffle = False,
        )
        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} val examples".format(len(val_data)))
        print("Loaded {} test examples".format(len(test_data)))

class CharCNNDataset(Dataset):
    def __init__(self, config):
        super(CharCNNDataset, self).__init__(config)
    
    def load_data(self, train_file, test_file, val_file=None):
        text_preprocessor = clean_text
        tokenizer = lambda sent: list(text_preprocessor(sent[::-1])) #reverse and split sent to List(char)
        TEXT = data.Field(sequential = True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        train_datafields = [('text',TEXT),('label',LABEL)]
        test_datafields = [('text',TEXT)]

        #load train and testt data into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file, mode='train')
        train_examples = [data.Example.fromlist(i, train_datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, train_datafields)

        test_df = self.get_pandas_df(test_file, mode='test')
        test_examples = [data.Example.fromlist(i, test_datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, test_datafields)

        #if validation file exists, then load in the train way; otherwise spilt train_data
        if val_file:
            val_df = self.get_pandas_df(val_file, mode='val')
            val_examples = [data.Example.fromlist(i, train_datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, train_datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)
        
        TEXT.build_vocab(train_data)
        embedding_mat = get_embedding_matrix(TEXT.vocab.stoi)
        TEXT.vocab.set_vectors(TEXT.vocab.stoi, torch.FloatTensor(embedding_mat), len(TEXT.vocab.stoi))
        self.vocab = TEXT.vocab
        self.char_embeddings = TEXT.vocab.vectors

        self.train_iterator = BucketIterator(
                                train_data, 
                                self.config.batch_size, 
                                sort_key=lambda x: len(x.text),
                                repeat = False,
                                shuffle= True)
        
        self.val_iterator, self.test_iterator = BucketIterator.splits(
            (val_data, test_data),
            batch_size = self.config.batch_size,
            sort_key = lambda x: len(x.text),
            repeat = False,
            shuffle = False,
        )
        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} val examples".format(len(val_data)))
        print("Loaded {} test examples".format(len(test_data)))


def evaluate(model, iterator):
    model.eval()
    total_preds = []
    total = []
    with torch.no_grad():
        for idx,batch in enumerate(iterator):
            if torch.cuda.is_available():
                x = batch.text.cuda()
            else:
                x = batch.text
            # x_tmp = torch.randint_like(x,high=255).cuda()
            output = model(x)
            # print(output)
            y_pred = (output.cpu().data.ge(0.5)+0).tolist()
            total_preds += y_pred 
            total += batch.label.cpu().data.tolist()
    acc = accuracy_score(total, total_preds)
    f1 = f1_score(total, total_preds)
    return acc, f1

def predict(model, iterator):
    model.eval()
    total_preds = []
    with torch.no_grad():
        for batch in iterator:
            if torch.cuda.is_available():
                x = batch.text.cuda()
            else:
                x = batch.text
            output = model(x)
            y_pred = (output.cpu().data.ge(0.5)+0).tolist()
            total_preds += y_pred
    
    test_df = pd.read_csv('./data/test.csv')
    test_id = test_df['id']
    sub_df = pd.DataFrame({"id":test_id, 'target':total_preds})
    sub_df.sort_values(by=['id'], axis=0, ascending=True, inplace=True)
    sub_df.to_csv('submission.csv', index=False)

def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def get_lr(scheduler):
    # for param_group in optimizer.param_groups:
    #     return param_group['lr']
    last_lr = scheduler.get_last_lr()[0]
    return last_lr

def get_lr_v1(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']