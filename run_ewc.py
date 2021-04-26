import torch
import argparse
from avalanche.benchmarks import PermutedMNIST, SplitMNIST
from avalanche.training.strategies import EWC
from avalanche.models import SimpleMLP
from avalanche.evaluation.metrics import ExperienceForgetting, \
    accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


import const
import json
from dataset import generate_rumour_scenario
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModel

"""
This example tests EWC on Split MNIST and Permuted MNIST.
It is possible to choose, among other options, between EWC with separate
penalties and online EWC with a single penalty.

On Permuted MNIST EWC maintains a very good performance on previous tasks
with a wide range of configurations. The average accuracy on previous tasks
at the end of training on all task is around 85%,
with a comparable training accuracy.

On Split MNIST, on the contrary, EWC is not able to remember previous tasks and
is subjected to complete forgetting in all configurations. The training accuracy
is above 90% but the average accuracy on previou tasks is around 20%.
"""

def main(args):

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
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(
            minibatch=True, epoch=True, experience=True, stream=True),
        ExperienceForgetting(),
        loggers=[interactive_logger])

    encoder = AutoModel.from_pretrained(model_name, config=config)
    encoder.eval()
    encoder.to(device)
    
    for version, event_names in zip(['d2'], [const.events_d2]):
    # for version, event_names in zip(['d1'], [const.events_d1]):
        with open("log/ewc_{}.log".format(args.custom_name), 'a') as out_file:
            out_file.write(version)
            out_file.write("\n")
            out_file.write("lambda:{}, epoch:{}, lr:{} \n".format(args.ewc_lambda, args.epochs, args.lr))


            # PREPARE SCENARIO FOR CL
            scenario = generate_rumour_scenario(tokenizer, event_names, args.dataset_name, max_len=128)


            if args.only_mlp:
                model = SimpleMLP(num_classes=3, input_size=768, hidden_size=512)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            criterion = torch.nn.CrossEntropyLoss()

            # CREATE THE STRATEGY
            strategy = EWC(model, optimizer, criterion, args.ewc_lambda,
                        args.ewc_mode, decay_factor=args.decay_factor,
                        train_epochs=args.epochs, device=device,
                        train_mb_size=args.train_minibatch_size, 
                        eval_mb_size=args.eval_minibatch_size,
                        evaluator=eval_plugin,
                        only_mlp=args.only_mlp, 
                        encoder=encoder
                        )

            # train on the selected scenario with the chosen strategy
            print('Starting experiment...')
            for experience in scenario.train_stream:
                print("Start training on experience ", experience.current_experience)

                strategy.train(experience)
                print("End training on experience", experience.current_experience)

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
    
    parser.add_argument('--ewc_mode', type=str, choices=['separate', 'online'],
                        default='separate', help='Choose between EWC and online.')
    parser.add_argument('--ewc_lambda', type=float, default=0.4,
                        help='Penalty hyperparameter for EWC - magnitude/importance of EWC regularization')
    parser.add_argument('--decay_factor', type=float, default=0.1,
                        help='Decay factor for importance when ewc_mode is online.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--train_minibatch_size', type=int, default=4,
                        help='Train minibatch size.')
    parser.add_argument('--eval_minibatch_size', type=int, default=4,
                        help='Eval minibatch size.')

    parser.add_argument('--only_mlp', action='store_true')
    parser.add_argument('--custom_name', type=str, default='', help='Custom Experiment Name')
    parser.add_argument('--use_topic_token', action='store_true')
    parser.add_argument('--dataset_name', type=str, default="topics", help='Name of the dataset')
    

    parser.add_argument('--cuda', type=int, default=0,
                        help='Specify GPU id to use. Use CPU if -1.')
    args = parser.parse_args()

    main(args)