import torch
import argparse
from avalanche.benchmarks import PermutedMNIST, SplitMNIST
from avalanche.training.strategies import GEM, AGEM
from avalanche.models import SimpleMLP
from avalanche.evaluation.metrics import ExperienceForgetting, accuracy_metrics, \
    loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

import const
import json
from dataset import generate_rumour_scenario
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModel

"""
This example tests both GEM and A-GEM on Split MNIST and Permuted MNIST.
GEM is a streaming strategy, that is it uses only 1 training epochs.
A-GEM may use a larger number of epochs.
Both GEM and A-GEM work with small mini batches (usually with 10 patterns).

Warning1: This implementation of GEM and A-GEM does not use task vectors.
Warning2: GEM is much slower than A-GEM.

Results (learning rate is always 0.1):

GEM-PMNIST (5 experiences):
Hidden size 512. 1 training epoch. 512 patterns per experience, 0.5 memory 
strength. Average Accuracy over all experiences at the end of training on the 
last experience: 92.6%

GEM-SMNIST:
Patterns per experience: 256, Memory strength: 0.5, hidden size: 256
Average Accuracy over all experiences at the end of training on the last 
experience: 93.3%

AGEM-PMNIST (5 experiences):
Patterns per experience = sample size: 256. 256 hidden size, 1 training epoch.
Average Accuracy over all experiences at the end of training on the last 
experience: 51.4%

AGEM-SMNIST:
Patterns per experience = sample size: 256, 512, 1024. Performance on previous tasks
remains very bad in terms of forgetting. Training epochs do not change result.
Hidden size 256.
Results for 1024 patterns per experience and sample size, 1 training epoch.
Average Accuracy over all experiences at the end of training on the last 
experience: 23.5%

"""


def main(args):
    # model = SimpleMLP(hidden_size=args.hs)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # criterion = torch.nn.CrossEntropyLoss()

    # check if selected GPU is available or use CPU
    assert args.cuda == -1 or args.cuda >= 0, "cuda must be -1 or >= 0."
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")
    print(f'Using device: {device}')

    model_name = 'bert-base-uncased'
    config = AutoConfig.from_pretrained(model_name, num_labels=3, return_dict=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        ExperienceForgetting(),
        loggers=[interactive_logger])
    
    encoder = AutoModel.from_pretrained(model_name, config=config)
    encoder.eval()
    encoder.to(device)

    # for version, event_names in zip(["v3"], [const.events_v3]):
    for version, event_names in zip(["v4"], [const.events_v4]):
        with open("log/gem_{}.log".format(args.custom_name), 'a') as out_file:
            out_file.write(version)
            out_file.write("\n")
            out_file.write("lambda:{}, epoch:{}, lr:{} \n".format(args.memory_strength, args.epochs, args.lr))

            # PREPARE SCENARIO FOR CL
            scenario = generate_rumour_scenario(tokenizer, event_names, args.dataset_name, max_len=128)
            
            if args.only_mlp:
                model = SimpleMLP(num_classes=3, input_size=768, hidden_size=512)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            criterion = torch.nn.CrossEntropyLoss()


            # create strategy
            if args.strategy == 'gem':
                strategy = GEM(model, optimizer, criterion, args.patterns_per_exp,
                            args.memory_strength, train_epochs=args.epochs,
                            device=device, train_mb_size=10, evaluator=eval_plugin, 
                            only_mlp=args.only_mlp, encoder=encoder)
            elif args.strategy == 'agem':
                strategy = AGEM(model, optimizer, criterion, args.patterns_per_exp,
                                args.sample_size, train_epochs=args.epochs, device=device,
                                train_mb_size=10, evaluator=eval_plugin, only_mlp=args.only_mlp, 
                                encoder=encoder)
            else:
                raise ValueError("Wrong strategy name. Allowed gem, agem.")
            # train on the selected scenario with the chosen strategy
            print('Starting experiment...')

            for experience in scenario.train_stream:
                print("Start training on experience ", experience.current_experience)

                strategy.train(experience)
                print("End training on experience ", experience.current_experience)
                print('Computing accuracy on the test set')
                result = strategy.eval(scenario.test_stream)[0]

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
    parser.add_argument('--strategy', type=str, choices=['gem', 'agem'],
                        default='gem', help='Choose between GEM and A-GEM')
    parser.add_argument('--patterns_per_exp', type=int, default=256,
                        help='Patterns to store in the memory for each'
                             ' experience')
    parser.add_argument('--sample_size', type=int, default=256,
                        help='Number of patterns to sample from memory when \
                        projecting gradient. A-GEM only.')
    parser.add_argument('--memory_strength', type=float, default=0.5,
                        help='Offset to add to the projection direction. GEM only.')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate.')
    parser.add_argument('--hs', type=int, default=256, help='MLP hidden size.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs.')
    parser.add_argument('--permutations', type=int, default=5,
                        help='Number of experiences in Permuted MNIST.')
    
    parser.add_argument('--only_mlp', action='store_true')
    parser.add_argument('--custom_name', type=str, default='')
    parser.add_argument('--dataset_name', type=str, default="topics", help='Name of the dataset')

    parser.add_argument('--cuda', type=int, default=0,
                        help='Specify GPU id to use. Use CPU if -1.')
    args = parser.parse_args()

    main(args)