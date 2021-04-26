from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop
from torch.utils.data import Dataset, DataLoader, TensorDataset


from avalanche.benchmarks import nc_scenario
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.benchmarks.generators import ni_scenario, dataset_scenario


from transformers import AutoTokenizer, AutoModel
import const
import json

LABELS = {
    '0': 0, '1': 1
}

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None, task=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.task = task

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



class DataProcessor():
    def __init__(self, event_name, dataset_name):

        train_data_path = "/home/nayeon/Course_5214_Project/data/{}/{}.train.json".format(dataset_name, event_name)
        test_data_path = "/home/nayeon/Course_5214_Project/data/{}/{}.test.json".format(dataset_name, event_name)
        
        with open(train_data_path) as json_file:
            self.train = json.load(json_file)

        with open(test_data_path) as json_file:
            self.test = json.load(json_file)
        

    def get_train_examples(self):
        return self._create_examples(self.train, "train")

    def get_test_examples(self):
        return self._create_examples(self.test, "test")

    def get_labels(self):
        return [0,1]

    def _create_examples(self, objs, set_type):
        examples = []
        for (i, obj) in enumerate(objs):
            
            text_a = obj['sentence']
            label = obj['label']

            examples.append(
                InputExample(guid=i, text_a=text_a, text_b='', label=label))
        return examples


def convert_to_features(examples, 
                        tokenizer,
                        max_length=None,
                        label_map=None):

    if max_length is None:
        max_length = tokenizer.model_max_length
    
    labels = [label_map[str(example.label)] for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):

        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    return features


def get_dataset(event_name, tokenizer, dataset_name, phase='train', max_len=None):
    processor = DataProcessor(event_name, dataset_name)

    if phase == 'train':
        examples = processor.get_train_examples()
    else:
        examples = processor.get_test_examples()

    features = convert_to_features(examples,
                                    tokenizer,
                                    label_map=LABELS,
                                    max_length=max_len)
    

    all_input_triples = torch.tensor([(f.input_ids, f.attention_mask, f.token_type_ids) for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_triples, all_labels)

    return dataset

def generate_rumour_scenario(tokenizer, events, dataset_name, max_len=None):
    
    train_datasets = []
    test_datasets = []

    for event_name in events:

        train_dataset = get_dataset(event_name, tokenizer, dataset_name, 'train', max_len)
        test_dataset = get_dataset(event_name, tokenizer, dataset_name, 'test', max_len)

        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
        
    generic_scenario =  dataset_scenario(
        train_dataset_list=train_datasets,
        test_dataset_list=test_datasets,
        task_labels=list(range(len(events)))
    )

    return generic_scenario


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    generate_rumour_scenario(tokenizer)