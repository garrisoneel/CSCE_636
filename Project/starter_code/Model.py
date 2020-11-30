### YOUR CODE HERE
# import tensorflow as tf
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as tf
import os
import time
import tqdm
import numpy as np
from Network import ResNet
from DataLoader import MyDataset

"""This script defines the training, validation and testing process.
"""


class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        # self.network = MyNetwork(configs)
        # self.network = MyNetwork.ResNetBottleNeck(3)
        self.network = ResNet(configs['block'], configs['depth'])
        print(self.network)

    def model_setup(self):
        for m in self.network.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                # pass
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        #initialize model parameters
        self.model_setup()
        batch_size = configs['batch_size']
        max_epoch = configs['max_epoch']
        dataset = MyDataset(x_train, y_train, training=True)

        train_stats = {'epoch': [],
                       'bs': [],
                       'lr': [],
                       'loss': [],
                       'val_loss': [],
                       'val_score': [],
                       }

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        def set_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        def schedule_lr(epoch, config):
            keys = config.keys()
            for key in keys:
                if epoch < key:
                    return config[key]
            return config[key]

        torch.backends.cudnn.benchmark = True

        batchsize_schedule = [0, 100, 140, 160]
        iters = [1, 1, 1, 1]
        # iters = [1,5,25,125]
        batches_per_update = 1
        self.network.train()
        optimizer = torch.optim.SGD(self.network.parameters(),
        lr=configs["learning_rate"], weight_decay=configs['weight_decay'])  # initialize optimizer
        criterion = nn.CrossEntropyLoss()  # initialize loss function
        loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)

        best_epoch = 0
        best_val = 10
        self.network = self.network.cuda()
        # set_lr(optimizer,0.01)
        for _ in range(max_epoch):
            self.network.train()

            set_lr(optimizer, schedule_lr(_, configs['lr_schedule']))

            for b, bs in zip(batchsize_schedule, iters):
                if _ >= b:
                    new_bs = configs["batch_size"]*bs
                    if new_bs > batch_size:
                        batches_per_update = bs
                        batch_size = new_bs
                        print("batchsize {}".format(new_bs))

            train_stats['epoch'].append(_)
            train_stats['lr'].append(get_lr(optimizer))
            train_stats['bs'].append(batch_size)
            total_loss = 0

            its = 0
            ops = 0

            optimizer.zero_grad()
            for x_batch, y_batch in tqdm.tqdm(loader, desc="Epoch {}".format(_)):
                its += 1
                x_batch = x_batch.float().cuda()
                y_batch = y_batch.long().cuda()

                y_pred = self.network(x_batch)
                loss = criterion(y_pred, y_batch)
                loss = loss/batches_per_update
                total_loss += float(loss)
                loss.backward()
                if its % batches_per_update == 0 or its == x_batch.shape[0]:
                    optimizer.step()
                    optimizer.zero_grad()
                    ops += 1

            total_loss /= its
            print(total_loss)
            train_stats['loss'].append(total_loss)
            val_loss = 0
            # do validation
            if x_valid is not None and y_valid is not None:
                score, val_loss = self.evaluate(x_valid, y_valid)
                print("score = {:.3f}% ({:.4f}) in validation set.\n".format(
                    score*100, val_loss))

                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = _
                    print("Best loss yet!")
                    self.save(acc=score, epoch=_)
                else:
                    print("best was {} epochs ago".format(_-best_epoch))
                train_stats['val_loss'].append(val_loss)
                train_stats['val_score'].append(score)

        return train_stats

    def evaluate(self, x, y):
        self.network.eval()
        crit = nn.CrossEntropyLoss()
        dataset = MyDataset(x, y, training=False)
        loader = DataLoader(dataset, batch_size=min(1000, x.shape[0]))
        score = 0
        total_loss = 0
        with torch.no_grad():
            for batch, (x_sample, y_sample) in enumerate(loader):
                x_sample = x_sample.float().cuda()
                y_sample = y_sample.long().cuda()
                y_prob = self.network(x_sample)
                loss = crit(y_prob, y_sample)
                preds = torch.argmax(y_prob, dim=1)
                score += torch.eq(preds, y_sample).sum().item()
                total_loss += float(loss)
        score /= x.shape[0]
        total_loss /= batch
        return score, total_loss

    def predict_prob(self, x):  # predict class probabilities
        with torch.no_grad():
            probs = self.network(x.cuda())
        probs = F.softmax(probs, dim=1).cpu().numpy()
        return probs

    def save(self, acc=0, epoch=0):
        print("Saving...")
        chkpt = {
            'weights': self.network.state_dict(),
            'configs': self.configs,
            'acc': acc,
            'epoch': epoch,
        }
        path = os.path.abspath(self.configs['save_dir'])
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(chkpt, os.path.join(path, self.configs['name'] + '.ckpt'))

    def load(self):
        fn = os.path.join(self.configs['save_dir'],
                          self.configs['name'] + '.ckpt')
        chkpt = torch.load(fn)
        print("Loading from file: ")
        configs = chkpt['configs']
        print(configs)
        self.network = ResNet(configs['block'], configs['depth'])
        self.network.load_state_dict(chkpt['weights'])
        self.network.cuda()
        return

### END CODE HERE
