from datetime import datetime
import os
import typing

import torch.utils.data
import torch.optim
from torch.nn import functional as F
import tqdm

from training.utils import *


class Trainer:

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: str,
                 scheduler: torch.optim.lr_scheduler.LambdaLR, gradient_max_norm: float,
                 ema: EMA, save_frequency: int, checkpoints_dir: str, load_from_checkpoint: bool = False,
                 checkpoint_filepath: str = ""):

        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.save_frequency = save_frequency
        self.checkpoints_dir = checkpoints_dir
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        self.gradient_max_norm = gradient_max_norm
        self.scheduler = scheduler
        self.ema = ema

        self.init_epoch = 1
        self.steps_performed = 0
        self.best_em = 0
        self.best_f1 = 0

        if load_from_checkpoint:
            self.__load_checkpoint(checkpoint_filepath)
            self.model = self.model.to(self.device)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = state[k].to(self.device)

    def __load_checkpoint(self, checkpoint_filepath: str):
        checkpoint = torch.load(checkpoint_filepath)
        self.init_epoch = checkpoint['init_epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_f1 = checkpoint['best_f1']
        self.best_em = checkpoint['best_em']
        self.steps_performed = checkpoint['steps_performed']
        self.scheduler.last_epoch = self.init_epoch

    def __save_checkpoint(self, epoch: int, f1: float, em: float):
        self.ema.assign(self.model)
        state = {
            'init_epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'best_em': self.best_em,
            'steps_performed': self.steps_performed + 1,
        }

        filename = os.path.join(self.checkpoints_dir, 'epoch{:02d}_f1_{:.5f}_em_{:.5f}.pth.tar'.format(epoch, f1, em))
        print("Saving checkpoint:", filename, "...")
        torch.save(state, filename)
        self.ema.resume(self.model)

    def train(self, n_epochs: int, train_data_loader: torch.utils.data.DataLoader,
              validation_data_loader: torch.utils.data.DataLoader, validation_eval_dict: typing.Dict,
              perform_early_stopping: bool = False,
              patience: int = -1):
        elapsed_patience = 0
        for epoch in range(self.init_epoch, self.init_epoch + n_epochs):
            self.train_epoch(train_data_loader, epoch)
            self.ema.assign(self.model)
            print("Validating...")
            metrics = self.valid(validation_data_loader, validation_eval_dict)
            print(metrics)
            self.ema.resume(self.model)

            if perform_early_stopping and metrics["F1"] < self.best_f1 and metrics["EM"] < self.best_em:
                elapsed_patience += 1
                if elapsed_patience >= patience:
                    print("Early stopping...")
                    break

            if epoch % self.save_frequency == 0:
                self.__save_checkpoint(epoch, metrics["F1"], metrics["EM"])

            self.best_f1 = max(self.best_f1, metrics["F1"])
            self.best_em = max(self.best_em, metrics["EM"])

    def train_epoch(self, train_data_loader: torch.utils.data.DataLoader, epoch_num: int):
        avg_epoch_loss = 0.0
        n_losses = 0
        with torch.enable_grad(), tqdm.tqdm(total=len(train_data_loader.dataset)) as pbar:
            for cwidxs, ccidxs, qwidxs, qcidxs, ans_start, ans_end, qids in train_data_loader:
                cwidxs = cwidxs.to(self.device)
                ccidxs = ccidxs.to(self.device)
                qwidxs = qwidxs.to(self.device)
                qcidxs = qcidxs.to(self.device)
                ans_start = ans_start.to(self.device)
                ans_end = ans_end.to(self.device)

                batch_size = cwidxs.size(0)
                self.optimizer.zero_grad()

                # forward
                logit1, logit2 = self.model(cwidxs, ccidxs, qwidxs, qcidxs)
                loss = F.cross_entropy(logit1, ans_start) + F.cross_entropy(logit2, ans_end)
                loss_val = loss.item()
                avg_epoch_loss = (avg_epoch_loss * n_losses + loss_val) / (n_losses + 1)
                n_losses += 1

                # backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_max_norm)  # gradient clipping
                self.optimizer.step()  # model parameters update
                self.scheduler.step()  # lr update
                self.ema(self.model, self.steps_performed)  # model parameters ema

                pbar.update(batch_size)
                pbar.set_postfix(epoch=epoch_num, batch_ce=loss_val, avg_loss=avg_epoch_loss, learning_rate=self.scheduler.get_last_lr(),
                                 time=datetime.now().strftime('%b-%d_%H-%M'))
                self.steps_performed += 1

    def valid(self, data_loader: torch.utils.data.DataLoader, ground_truths: typing.Dict):
        self.model.eval()
        predictions = {}

        with torch.no_grad(), tqdm.tqdm(total=len(data_loader.dataset)) as pbar:
            for cwidxs, ccidxs, qwidxs, qcidxs, ans_start, ans_end, qids in data_loader:
                cwidxs = cwidxs.to(self.device)
                ccidxs = ccidxs.to(self.device)
                qwidxs = qwidxs.to(self.device)
                qcidxs = qcidxs.to(self.device)
                ans_start = ans_start.to(self.device)
                ans_end = ans_end.to(self.device)

                batch_size = cwidxs.size(0)

                logit1, logit2 = self.model(cwidxs, ccidxs, qwidxs, qcidxs)
                p1, p2 = F.softmax(logit1, dim=-1), F.softmax(logit2, dim=-1)

                loss = F.cross_entropy(logit1, ans_start) + F.cross_entropy(logit2, ans_end)
                loss_val = loss.item()

                pred_ans_start, pred_ans_end = get_answers_spans(p1, p2)

                prediction = get_predictions_from_spans(ground_truths, qids, pred_ans_start, pred_ans_end)
                predictions.update(prediction)

                pbar.update(batch_size)
                pbar.set_postfix(batch_ce=loss_val, time=datetime.now().strftime('%b-%d_%H-%M'))

        self.model.train()

        metrics = compute_metrics(ground_truths, predictions)
        return metrics
