################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
from collections import defaultdict
from typing import Optional, Sequence, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer

from avalanche.benchmarks.scenarios import Experience
from avalanche.benchmarks.utils.data_loader import \
    MultiTaskMultiBatchDataLoader, MultiTaskDataLoader
from avalanche.logging import default_logger
from typing import TYPE_CHECKING

from avalanche.training.plugins import EvaluationPlugin

if TYPE_CHECKING:
    from avalanche.training.plugins import StrategyPlugin

from transformers import AutoConfig, AutoModel

class BaseStrategy:
    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence['StrategyPlugin']] = None,
                 evaluator=default_logger, eval_every=-1, only_mlp=False, encoder=None):
        """
        BaseStrategy is the super class of all task-based continual learning
        strategies. It implements a basic training loop and callback system
        that allows to execute code at each experience of the training loop.
        Plugins can be used to implement callbacks to augment the training
        loop with additional behavior (e.g. a memory buffer for replay).

        **Scenarios**
        This strategy supports several continual learning scenarios:

        * class-incremental scenarios (no task labels)
        * multi-task scenarios, where task labels are provided)
        * multi-incremental scenarios, where the same task may be revisited

        The exact scenario depends on the data stream and whether it provides
        the task labels.

        **Training loop**
        The training loop and its callbacks are organized as follows::
            train
                before_training
                before_training_exp
                adapt_train_dataset
                make_train_dataloader
                before_training_epoch
                    before_training_iteration
                        before_forward
                        after_forward
                        before_backward
                        after_backward
                    after_training_iteration
                    before_update
                    after_update
                after_training_epoch
                after_training_exp
                after_training

        **Evaluation loop**
        The evaluation loop and its callbacks are organized as follows::
            eval
                before_eval
                adapt_eval_dataset
                make_eval_dataloader
                before_eval_exp
                    eval_epoch
                        before_eval_iteration
                        before_eval_forward
                        after_eval_forward
                        after_eval_iteration
                after_eval_exp
                after_eval

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop.
                if -1: no evaluation during training.
                if  0: calls `eval` after the final epoch of each training
                    experience.
                if >0: calls `eval` every `eval_every` epochs and at the end
                    of all the epochs for a single experience.
        """
        self.model: Module = model
        """ PyTorch model. """

        self.criterion = criterion
        """ Loss function. """

        self.optimizer = optimizer
        """ PyTorch optimizer. """

        self.train_epochs: int = train_epochs
        """ Number of training epochs. """

        self.train_mb_size: int = train_mb_size
        """ Training mini-batch size. """

        self.eval_mb_size: int = train_mb_size if eval_mb_size is None \
            else eval_mb_size
        """ Eval mini-batch size. """

        self.device = device
        """ PyTorch device where the model will be allocated. """

        self.plugins = [] if plugins is None else plugins
        """ List of `StrategyPlugin`s. """

        if evaluator is None:
            evaluator = EvaluationPlugin()
        self.plugins.append(evaluator)
        self.evaluator = evaluator
        """ EvaluationPlugin used for logging and metric computations. """

        self.eval_every = eval_every
        """ Frequency of the evaluation during training. """

        ###################################################################
        # State variables. These are updated during the train/eval loops. #
        ###################################################################
        self.training_exp_counter = 0
        """ Counts the number of training steps. +1 at the end of each 
        experience. """

        self.epoch: Optional[int] = None
        """ Epoch counter. """

        self.experience = None
        """ Current experience. """

        self.adapted_dataset = None
        """ Data used to train. It may be modified by plugins. Plugins can 
        append data to it (e.g. for replay). 
         
        .. note:: 
            This dataset may contain samples from different experiences. If you 
            want the original data for the current experience  
            use :attr:`.BaseStrategy.experience`.
        """

        self.current_dataloader = None
        """ Dataloader. """

        self.mb_it = None
        """ Iteration counter. Reset at the start of a new epoch. """

        self.mb_x = None
        """ Current mini-batch input. """

        self.mb_y = None
        """ Current mini-batch target. """

        self.loss = None
        """ Loss of the current mini-batch. """

        self.logits = None
        """ Logits computed on the current mini-batch. """

        self.is_training: bool = False
        """ True if the strategy is in training mode. """
        
        self.only_mlp = only_mlp
        """ Whether to apply CL strategy on the WHOLE model or only on the MLP layer """
        if self.only_mlp:
            self.encoder = encoder


    @property
    def is_eval(self):
        """ True if the strategy is in evaluation mode. """
        return not self.is_training

    def update_optimizer(self, old_params, new_params, reset_state=True):
        """ Update the optimizer by substituting old_params with new_params.

        :param old_params: List of old trainable parameters.
        :param new_params: List of new trainable parameters.
        :param reset_state: Wheter to reset the optimizer's state.
            Defaults to True.
        :return:
        """
        for old_p, new_p in zip(old_params, new_params):
            found = False
            # iterate over group and params for each group.
            for group in self.optimizer.param_groups:
                for i, curr_p in enumerate(group['params']):
                    if hash(curr_p) == hash(old_p):
                        # update parameter reference
                        group['params'][i] = new_p
                        found = True
                        break
                if found:
                    break
            if not found:
                raise Exception(f"Parameter {old_params} not found in the "
                                f"current optimizer.")
        if reset_state:
            # State contains parameter-specific information.
            # We reset it because the model is (probably) changed.
            self.optimizer.state = defaultdict(dict)

    def add_new_params_to_optimizer(self, new_params):
        """ Add new parameters to the trainable parameters.

        :param new_params: list of trainable parameters
        """
        self.optimizer.add_param_group({'params': new_params})

    def train(self, experiences: Union[Experience, Sequence[Experience]],
              eval_streams: Optional[Sequence[Union[Experience,
                                                    Sequence[
                                                        Experience]]]] = None,
              **kwargs):
        """ Training loop. if experiences is a single element trains on it.
        If it is a sequence, trains the model on each experience in order.
        This is different from joint training on the entire stream.

        :param experiences: single Experience or sequence.
        :param eval_streams: list of streams for evaluation.
            If None: use training experiences for evaluation.
            Use [] if you do not want to evaluate during training.
        """
        self.is_training = True
        self.model.train()
        self.model.to(self.device)

        # Normalize training and eval data.
        if isinstance(experiences, Experience):
            experiences = [experiences]
        if eval_streams is None:
            eval_streams = [experiences]
        for i, exp in enumerate(eval_streams):
            if isinstance(exp, Experience):
                eval_streams[i] = [exp]

        res = []
        self.before_training(**kwargs)
        for exp in experiences:
            self.train_exp(exp, eval_streams, **kwargs)
            res.append(self.evaluator.current_metrics.copy())

        self.after_training(**kwargs)
        return res

    def train_exp(self, experience: Experience, eval_streams, **kwargs):
        """
        Training loop over a single Experience object.

        :param experience: CL experience information.
        :param kwargs: custom arguments.
        """

        self.experience = experience
        self.model.train()
        self.model.to(self.device)

        self.adapted_dataset = experience.dataset
        self.adapted_dataset = self.adapted_dataset.train()
        self.adapt_train_dataset(**kwargs)
        self.make_train_dataloader(**kwargs)

        self.before_training_exp(**kwargs)
        self.epoch = 0
        for self.epoch in range(self.train_epochs):
            self.before_training_epoch(**kwargs)
            self.training_epoch(**kwargs)
            self._periodic_eval(eval_streams, do_final=False)
            self.after_training_epoch(**kwargs)
        self._periodic_eval(eval_streams, do_final=True)  # Final evaluation
        self.after_training_exp(**kwargs)

    def _periodic_eval(self, eval_streams, do_final):
        """ Periodic eval controlled by `self.eval_every`. """
        # Since we are switching from train to eval model inside the training
        # loop, we need to save the training state, and restore it after the
        # eval is done.
        _prev_state = (
            self.epoch,
            self.experience,
            self.adapted_dataset,
            self.current_dataloader)

        if self.eval_every == 0 and do_final:
            
            # we are outside the epoch loop
            for stream in eval_streams:
                self.eval(stream)
        elif self.eval_every > 0 and self.epoch % self.eval_every == 0:
            for stream in eval_streams:
                self.eval(stream)

        # restore train-state variables and training mode.
        self.epoch, self.experience, self.adapted_dataset = _prev_state[:3]
        self.current_dataloader = _prev_state[3]
        self.model.train()

    def eval(self,
             exp_list: Union[Experience, Sequence[Experience]],
             **kwargs):
        """
        Evaluate the current model on a series of experiences.

        :param exp_list: CL experience information.
        :param kwargs: custom arguments.
        """
        self.is_training = False
        self.model.eval()
        self.model.to(self.device)

        if isinstance(exp_list, Experience):
            exp_list = [exp_list]

        res = []
        self.before_eval(**kwargs)
        for exp in exp_list:
            self.experience = exp
            self.adapted_dataset = exp.dataset
            self.adapted_dataset = self.adapted_dataset.eval()

            self.adapt_eval_dataset(**kwargs)
            self.make_eval_dataloader(**kwargs)

            self.before_eval_exp(**kwargs)
            self.eval_epoch(**kwargs)
            self.after_eval_exp(**kwargs)
            
        res.append(self.evaluator.current_metrics.copy())

        self.after_eval(**kwargs)
        return res

    def before_training_exp(self, **kwargs):
        """
        Called  after the dataset and data loader creation and
        before the training loop.
        """
        for p in self.plugins:
            p.before_training_exp(self, **kwargs)

    def make_train_dataloader(self, num_workers=0, shuffle=True, **kwargs):
        """
        Called after the dataset instantiation. Initialize the data loader.
        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        """
        self.current_dataloader = MultiTaskMultiBatchDataLoader(
            self.adapted_dataset,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            shuffle=shuffle)

    def make_eval_dataloader(self, num_workers=0, **kwargs):
        """
        Initialize the eval data loader.
        :param num_workers:
        :param kwargs:
        :return:
        """
        self.current_dataloader = MultiTaskDataLoader(
            self.adapted_dataset,
            oversample_small_tasks=False,
            num_workers=num_workers,
            batch_size=self.eval_mb_size,
        )

    def adapt_train_dataset(self, **kwargs):
        """
        Called after the dataset initialization and before the
        dataloader initialization. Allows to customize the dataset. TODO: customize nayeon
        :param kwargs:
        :return:
        """
        self.adapted_dataset = {
            self.experience.task_label: self.adapted_dataset}
        for p in self.plugins:
            p.adapt_train_dataset(self, **kwargs)

    def before_training_epoch(self, **kwargs):
        """
        Called at the beginning of a new training epoch.
        :param kwargs:
        :return:
        """
        for p in self.plugins:
            p.before_training_epoch(self, **kwargs)

    # OG
    def process_input_for_bert(self, mb_x):
        input_ids, attention_mask, token_type_ids = zip(*mb_x)
        
        input_ids = torch.stack(input_ids, dim=0).to(self.device)
        attention_mask = torch.stack(attention_mask, dim=0).to(self.device)
        token_type_ids = torch.stack(token_type_ids, dim=0).to(self.device)

        return (input_ids, attention_mask, token_type_ids)
    
    def process_input_for_mlp_only (self, mb_x):
        input_ids, attention_mask, token_type_ids = zip(*mb_x)
        
        input_ids = torch.stack(input_ids, dim=0).to(self.device)
        attention_mask = torch.stack(attention_mask, dim=0).to(self.device)
        token_type_ids = torch.stack(token_type_ids, dim=0).to(self.device)

        outputs = self.encoder(input_ids, attention_mask, token_type_ids)

        pooled_output = outputs[1] 
        # print(outputs[0].shape) # bsz, 128, 768
        # print(outputs[1].shape) # bsz, 768

        return pooled_output
        

    def training_epoch(self, **kwargs):
        """
        Training epoch.
        :param kwargs:
        :return:
        """
        for self.mb_it, batches in \
                enumerate(self.current_dataloader):
            self.before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0
            for self.mb_task_id, (self.mb_x, self.mb_y) in batches.items():
                
                if self.only_mlp:
                    self.mb_x = self.process_input_for_mlp_only(self.mb_x)
                else:
                    self.mb_x = self.process_input_for_bert(self.mb_x)
                
                self.mb_y = self.mb_y.to(self.device)

                # Forward
                self.before_forward(**kwargs)
                if self.only_mlp:
                    self.logits = self.model(self.mb_x)
                else:
                    self.logits = self.model(*self.mb_x)[0]
                self.after_forward(**kwargs)

                # Loss & Backward
                self.loss += self.criterion(self.logits, self.mb_y)

            self.before_backward(**kwargs)
            self.loss.backward()
            self.after_backward(**kwargs)

            # Optimization step
            self.before_update(**kwargs)
            self.optimizer.step()
            self.after_update(**kwargs)

            self.after_training_iteration(**kwargs)

    def before_training(self, **kwargs):
        for p in self.plugins:
            p.before_training(self, **kwargs)

    def after_training(self, **kwargs):
        for p in self.plugins:
            p.after_training(self, **kwargs)

    def before_training_iteration(self, **kwargs):
        for p in self.plugins:
            p.before_training_iteration(self, **kwargs)

    def before_forward(self, **kwargs):
        for p in self.plugins:
            p.before_forward(self, **kwargs)

    def after_forward(self, **kwargs):
        for p in self.plugins:
            p.after_forward(self, **kwargs)

    def before_backward(self, **kwargs):
        for p in self.plugins:
            p.before_backward(self, **kwargs)

    def after_backward(self, **kwargs):
        for p in self.plugins:
            p.after_backward(self, **kwargs)

    def after_training_iteration(self, **kwargs):
        for p in self.plugins:
            p.after_training_iteration(self, **kwargs)

    def before_update(self, **kwargs):
        for p in self.plugins:
            p.before_update(self, **kwargs)

    def after_update(self, **kwargs):
        for p in self.plugins:
            p.after_update(self, **kwargs)

    def after_training_epoch(self, **kwargs):
        for p in self.plugins:
            p.after_training_epoch(self, **kwargs)

    def after_training_exp(self, **kwargs):
        for p in self.plugins:
            p.after_training_exp(self, **kwargs)

        self.training_exp_counter += 1
        # Reset flow-state variables. They should not be used outside the flow
        self.epoch = None
        self.experience = None
        self.adapted_dataset = None
        self.current_dataloader = None
        self.mb_it = None
        self.mb_it, self.mb_x, self.mb_y = None, None, None
        self.loss = None
        self.logits = None

    def before_eval(self, **kwargs):
        for p in self.plugins:
            p.before_eval(self, **kwargs)

    def before_eval_exp(self, **kwargs):
        for p in self.plugins:
            p.before_eval_exp(self, **kwargs)

    def adapt_eval_dataset(self, **kwargs):
        self.adapted_dataset = {
            self.experience.task_label: self.adapted_dataset}
        for p in self.plugins:
            p.adapt_eval_dataset(self, **kwargs)

    def eval_epoch(self, **kwargs):
        for self.mb_it, (self.mb_task_id, self.mb_x, self.mb_y) in \
                enumerate(self.current_dataloader):
            self.before_eval_iteration(**kwargs)
            
            if self.only_mlp:
                self.mb_x = self.process_input_for_mlp_only(self.mb_x)
            else:
                self.mb_x = self.process_input_for_bert(self.mb_x)
            
            self.mb_y = self.mb_y.to(self.device)

            self.before_eval_forward(**kwargs)

            if self.only_mlp:
                self.logits = self.model(self.mb_x)
            else:
                self.logits = self.model(*self.mb_x)[0]

            self.after_eval_forward(**kwargs)
            self.loss = self.criterion(self.logits, self.mb_y)

            self.after_eval_iteration(**kwargs)

    def after_eval_exp(self, **kwargs):
        for p in self.plugins:
            p.after_eval_exp(self, **kwargs)

    def after_eval(self, **kwargs):
        for p in self.plugins:
            p.after_eval(self, **kwargs)
        # Reset flow-state variables. They should not be used outside the flow
        self.experience = None
        self.adapted_dataset = None
        self.current_dataloader = None
        self.mb_it = None
        self.mb_it, self.mb_x, self.mb_y = None, None, None
        self.loss = None
        self.logits = None

    def before_eval_iteration(self, **kwargs):
        for p in self.plugins:
            p.before_eval_iteration(self, **kwargs)

    def before_eval_forward(self, **kwargs):
        for p in self.plugins:
            p.before_eval_forward(self, **kwargs)

    def after_eval_forward(self, **kwargs):
        for p in self.plugins:
            p.after_eval_forward(self, **kwargs)

    def after_eval_iteration(self, **kwargs):
        for p in self.plugins:
            p.after_eval_iteration(self, **kwargs)


__all__ = ['BaseStrategy']
