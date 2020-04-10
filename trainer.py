import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from itertools import repeat, chain, islice
import os

from config import ARGS
from network.util_network import ScheduledOptim, NoamOpt


class NoamOptimizer:
    def __init__(self, model, lr, model_size, warmup):
        self._adam = torch.optim.Adam(model.parameters(), lr=lr)
        self._opt = NoamOpt(
            model_size=model_size, factor=1, warmup=warmup, optimizer=self._adam)

    def step(self, loss):
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()


class Trainer:
    def __init__(self, model, device, warm_up_step_count,
                 d_model, num_epochs, weight_path, lr,
                 train_data, val_data, test_data):
        self._device = device
        self._num_epochs = num_epochs
        self._weight_path = weight_path

        self._model = model
        self._loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self._model.to(device)

        self._train_data = train_data
        self._val_data = val_data
        self._test_data = test_data

        self._opt = NoamOptimizer(model=model, lr=lr, model_size=d_model, warmup=warm_up_step_count)

        self.step = 0
        self._threshold = 0.5
        self.max_step = 0
        self.max_acc = 0
        self.max_auc = 0

        self.test_acc = 0
        self.test_auc = 0

    # train model and choose weight with max auc on validation dataset
    def train(self):
        train_gen = data.DataLoader(
            dataset=self._train_data, shuffle=True,
            batch_size=ARGS.train_batch, num_workers=ARGS.num_workers)
        val_gen = data.DataLoader(
            dataset=self._val_data, shuffle=False,
            batch_size=ARGS.test_batch, num_workers=ARGS.num_workers)

        # will train self._num_epochs copies of train data
        to_train = chain.from_iterable(repeat(train_gen, self._num_epochs))
        # consisting of total_steps batches
        total_steps = len(train_gen) * self._num_epochs
        print(total_steps)

        self.step = 0
        while self.step < total_steps:
            rem_steps = total_steps - self.step
            num_steps = min(rem_steps, ARGS.eval_steps)
            self.step += num_steps

            # take num_steps batches from to_train stream
            train_batches = islice(to_train, num_steps)
            print(f'Step: {self.step}')
            self._train(train_batches, num_steps)

            cur_weight = self._model.state_dict()
            torch.save(cur_weight, f'{self._weight_path}{self.step}.pt')
            self._test('Validation', val_gen)
            print(f'Current best weight: {self.max_step}.pt, best auc: {self.max_auc:.4f}')
            # remove all weight file except {self.max_step}.pt
            weight_list = os.listdir(self._weight_path)
            for w in weight_list:
                if int(w[:-3]) != self.max_step:
                    os.unlink(f'{self._weight_path}{w}')

    # get test results
    def test(self, weight_num):
        test_gen = data.DataLoader(
            dataset=self._test_data, shuffle=False,
            batch_size=ARGS.test_batch, num_workers=ARGS.num_workers)

        # load best weight
        if self.max_step != 0:
            weight_num = self.max_step
        weight_path = f'{ARGS.weight_path}{weight_num}.pt'
        print(f'best weight: {weight_path}')
        self._model.load_state_dict(torch.load(weight_path))
        self._test('Test', test_gen)

    def _forward(self, batch):
        batch = {k: t.to(self._device) for k, t in batch.items()}
        label = batch['label']  # shape: (batch_size, 1)

        output = self._model(batch['input'], batch['target_id'])
        pred = (torch.sigmoid(output) >= self._threshold).long()  # shape: (batch_size, 1)

        return label, output, pred

    def _get_loss(self, label, output):
        loss = self._loss_fn(output, label.float())
        return loss.mean()

    # takes iterator
    def _train(self, batch_iter, num_batches):
        start_time = time.time()
        self._model.train()

        losses = []
        num_corrects = 0
        num_total = 0
        labels = []
        outs = []

        for batch in tqdm(batch_iter, total=num_batches):
            label, out, pred = self._forward(batch)
            train_loss = self._get_loss(label, out)
            losses.append(train_loss.item())

            self._opt.step(train_loss)

            num_corrects += (pred == label).sum().item()
            num_total += len(label)

            labels.extend(label.squeeze(-1).data.cpu().numpy())
            outs.extend(out.squeeze(-1).data.cpu().numpy())


        acc = num_corrects / num_total
        auc = roc_auc_score(labels, outs)
        loss = np.mean(losses)
        training_time = time.time() - start_time


        print(f'correct: {num_corrects}, total: {num_total}')
        print(f'[Train]     time: {training_time:.2f}, loss: {loss:.4f}, acc: {acc:.4f}, auc: {auc:.4f}')

        # wandb.log({'Train acc': acc, 'Train loss': loss}, step=self.step)

    # takes iterable
    def _test(self, name, batches):
        start_time = time.time()
        self._model.eval()

        losses = []
        num_corrects = 0
        num_total = 0
        labels = []
        outs = []

        with torch.no_grad():
            for batch in tqdm(batches):
                label, out, pred = self._forward(batch)
                test_loss = self._get_loss(label, out)
                losses.append(test_loss.item())

                num_corrects += (pred == label).sum().item()
                num_total += len(label)

                labels.extend(label.squeeze(-1).data.cpu().numpy())
                outs.extend(out.squeeze(-1).data.cpu().numpy())

        acc = num_corrects / num_total
        auc = roc_auc_score(labels, outs)
        loss = np.mean(losses)
        training_time = time.time() - start_time

        print(f'correct: {num_corrects}, total: {num_total}')
        print(f'[{name}]      time: {training_time:.2f}, loss: {loss:.4f}, acc: {acc:.4f}, auc: {auc:.4f}')

        if name == 'Validation':
            if self.max_auc < auc:
                self.max_auc = auc
                self.max_acc = acc
                self.max_step = self.step

        elif name == 'Test':
            self.test_acc = acc
            self.test_auc = auc

        # wandb.log({
        #     'Test acc': acc,
        #     'Test auc': auc,
        #     'Test loss': loss,
        #     'Best acc': self.max_acc,
        #     'Best auc': self.max_auc
        # }, step=self.step)

