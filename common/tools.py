import numpy as np
from PIL import Image

import time
import torch
import datetime
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

from networks.Lenet import Lenet
from networks.Resnet import ResNet18, ResNet34


# create different models for different datasets
def createModel(modelName, num_classes=10):
    if modelName == 'Lenet':
        print('Building new Lenet(' + str(num_classes) + ')')
        model = Lenet()
    elif modelName == 'ResNet18':
        print('Building new ResNet18(' + str(num_classes) + ')')
        model = ResNet18(num_classes)
    elif modelName == 'ResNet34':
        print('Building new ResNet34(' + str(num_classes) + ')')
        model = ResNet34(num_classes)

    if torch.cuda.is_available():
        model.cuda()
    return model


# Evaluate models with a noisy validation
def evaluate_val(model, val_loader, num_classes, prefix):
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            logits = model(images)
            
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)

            for i in range(labels.size(0)):
                class_total[labels[i]] += 1
                if(pred[i] == labels[i]):
                    class_correct[labels[i]] += 1

    # To overcome the imbalance of noisy validation
    acc = round(np.average(100 * class_correct / class_total), 2)
    std = round(np.std(100 * class_correct / class_total), 2)

    if prefix != "":
        print(getTime(), prefix, acc)

    return acc, std


def evaluate(model, eva_loader, ceriation, prefix):
    losses = AverageMeter('Loss', ':3.2f')
    top1 = AverageMeter('Acc@1', ':3.2f')

    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(eva_loader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            logist = model(images)

            loss = ceriation(logist, labels)
            acc1, acc5 = accuracy(logist, labels, topk=(1, 5))

            losses.update(loss.item(), images[0].size(0))
            top1.update(acc1[0], images[0].size(0))

    if prefix != "":
        print(getTime(), prefix, round(top1.avg.item(), 2))

    return top1.avg.to("cpu", torch.float).item(), losses.avg


def predict(train_loader, model):
    model.eval()
    preds = np.array([])

    for images, labels in train_loader:
        if torch.cuda.is_available():
            images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred1 = torch.max(outputs.data, 1)
        preds = np.concatenate((preds, pred1.to("cpu", torch.int).numpy()), axis=0)

    return preds.astype(int).tolist()


def train(model, train_loader, optimizer, ceriation, epoch):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    losses = AverageMeter('Loss', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Train Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        logist = model(images)
        loss = ceriation(logist, labels)

        acc1, acc5 = accuracy(logist, labels, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    progress.display(0)
    return losses.avg, top1.avg.to("cpu", torch.float).item()


def isSame(preds, noise_labels, clean_labels, images, num_classes):
    labels = []
    data = []
    correct_number = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    for i in range(len(preds)):
        if(preds[i] == noise_labels[i]):
            data.append(images[i])
            labels.append(preds[i])
            class_total[preds[i]] += 1

            if clean_labels is not None:
                if(preds[i] == clean_labels[i]):
                    correct_number += 1
                    class_correct[preds[i]] += 1

    if clean_labels is not None:
        accracy = round(100 * correct_number / len(labels), 2)
        print('Same labels number:', len(labels), accracy)

        for i in range(num_classes):
            if(class_total[i] > 0):
                print('Accuracy of %5s : %.2f %%' % (i, 100 * class_correct[i] / class_total[i]))

    return data, labels


class Train_Dataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.train_data = np.array(data)
        self.train_labels = np.array(labels)
        self.length = len(self.train_labels)
        self.target_transform = target_transform

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.train_data, self.train_labels


def getTime():
    time_stamp = datetime.datetime.now()
    return time_stamp.strftime('%H:%M:%S')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def write_logs(title, args, best_test_acc):
    f = open("./logs/results.txt", "a")
    if args is not None:
        f.write("\n" + getTime() + " " + str(args) + "\n")
    f.write(getTime() + " " + title + " seed-" + str(args.seed) + ", Best Test Acc: " + str(best_test_acc) + "\n")
    f.close()
