################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-10-2020                                                             #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a simple example on how to use the Replay strategy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop
import torch.optim.lr_scheduler
from avalanche.benchmarks import nc_scenario
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.evaluation.metrics import ExperienceForgetting, \
    accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

import const
import json
from dataset import generate_rumour_scenario
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def main(args):
    # --- CONFIG
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")
    n_batches = 5
    

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        ExperienceForgetting(),
        loggers=[interactive_logger])

    # ---------
    

    replay_size = 25

    model_name = 'bert-base-uncased'
    config = AutoConfig.from_pretrained(model_name, num_labels=3, return_dict=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    for version, event_names in zip(['d2'], [const.events_d2]):
    # for version, event_names in zip(['d1'], [const.events_d1]):
        with open("log/replay_{}.log".format(replay_size), 'a') as out_file:
            out_file.write(version)
            out_file.write("\n")

            # --- SCENARIO CREATION
            scenario = generate_rumour_scenario(tokenizer, event_names, args.dataset_name, max_len=128)
            # ---------

            # MODEL CREATION
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # CREATE THE STRATEGY INSTANCE (NAIVE) with REPLAY PLUGIN
            cl_strategy = Naive(model, optimizer,
                                CrossEntropyLoss(),
                                train_mb_size=4, train_epochs=args.epochs, eval_mb_size=4, device=device,
                                plugins=[ReplayPlugin(mem_size=replay_size)],
                                evaluator=eval_plugin
                                )

            # TRAINING LOOP
            print('Starting experiment...')
            for experience in scenario.train_stream:
                print("Start of experience ", experience.current_experience)
                cl_strategy.train(experience)
                print('Training completed')

                print('Computing accuracy on the whole test set')

                result = cl_strategy.eval(scenario.test_stream)[0]
                values = []
                for key, val in result.items():
                    if "Top1_Acc_Exp" in key:
                        values.append(str(val))
                
                result_str = ",".join(values)
                # out_file.write("Event {}\n".format(event_names[idx]))
                out_file.write(result_str)
                out_file.write("\n")
            
            out_file.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0,
                        help='Select zero-indexed cuda device. -1 to use CPU.')
    parser.add_argument('--use_topic_token', action='store_true')
    parser.add_argument('--dataset_name', type=str, default="topics", help='Name of the dataset')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs.')
    
    args = parser.parse_args()
    main(args)
