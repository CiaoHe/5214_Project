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
from torch.utils.data import Dataset, DataLoader

from avalanche.benchmarks import nc_scenario
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.benchmarks.generators import ni_scenario, dataset_scenario

import const
import json
from dataset import generate_rumour_scenario
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


model_path = "bert-base-uncased"

def main(args):
    # --- CONFIG
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")
    # ---------

    # For Naive BERT
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path, num_labels=3, return_dict=False)

    for version, event_names in zip(['d2'], [const.events_d2]):
    # for version, event_names in zip(['d1'], [const.events_d1]):
        with open("log/{}.log".format(args.log_f_name), 'a') as out_file:
            out_file.write("{}, epoch:{}, lr:{} \n".format(version, args.epochs, args.lr))

            # PREPARE SCENARIO FOR CL
            scenario = generate_rumour_scenario(tokenizer, event_names, args.dataset_name, max_len=128)

            # MODEL CREATION
            model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
            
            # CREATE THE STRATEGY INSTANCE (NAIVE)
            cl_strategy = Naive(
                model, SGD(model.parameters(), lr=args.lr, momentum=0.9),
                CrossEntropyLoss(), train_mb_size=4, train_epochs=args.epochs, eval_mb_size=4,
                device=device)

            # TRAINING LOOP
            print('Starting experiment...')
            results = []
            for idx, experience in enumerate(scenario.train_stream):
                print("Start of experience: ", experience.current_experience)
                print("Current Classes: ", experience.classes_in_this_experience)

                cl_strategy.train(experience)
                print('Training completed')

                print('Computing accuracy on the whole test set')
            
                result = cl_strategy.eval(scenario.test_stream)[0]

                values = []
                for key, val in result.items():
                    if "Top1_Acc_Exp" in key:
                        values.append(str(val))
                        # result_str += + str(value)
                
                result_str = ",".join(values)
                out_file.write(result_str)
                out_file.write("\n")
            out_file.write("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0,
                        help='Select zero-indexed cuda device. -1 to use CPU.')
    parser.add_argument('--use_topic_token', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs.')
    parser.add_argument('--log_f_name', default='naive', type=str, help="Name of the log file")
    parser.add_argument('--dataset_name', type=str, default="topics", help='Name of the dataset')

    args = parser.parse_args()

    main(args)
    # # main(args, const.events_for_update)