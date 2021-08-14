import os
import os.path
import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms

from common.tools import getTime, train, evaluate


np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
print("PyTorch version:", torch.__version__)
os.system('nvidia-smi')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, help='seed number', default=1)
parser.add_argument('--lr', type=float, help='initial learning rate', default=0.005)
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_classes', type=int, default=14)
parser.add_argument('--pretrain', action='store_false', help='pretrain')
parser.add_argument('--model_dir', type=str, help='dir to save model files', default='model')
parser.add_argument('--data_root', type=str, help='data location', default='data/Clothing1M_Official/')
parser.add_argument('--beta', type=float, help='beta for scores', default=0.45)
parser.add_argument('--n_epoch', type=int, default=5)
parser.add_argument('--max_inner_loop', type=int, help='max inner round', default=6)
parser.add_argument('--max_outer_loop', type=int, help='max outer round', default=3)
args = parser.parse_args()
print(args)


class Clothing1M_Dataset(Dataset):
    def __init__(self, data, labels, root_dir, transform=None):
        self.train_data = np.array(data)
        self.train_labels = np.array(labels)
        self.root_dir = root_dir
        self.length = len(self.train_labels)

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        # print("Dataset length:", self.length)

    def __getitem__(self, index):
        img_paths, target = self.train_data[index], self.train_labels[index]

        img_paths = os.path.join(self.root_dir, img_paths)
        img = Image.open(img_paths).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.train_data, self.train_labels


def createModel(pretrained):
    model = torchvision.models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(2048, args.num_classes)
    return model.cuda()


def evaluate_val(test_loader, model, loss_func):
    model.eval()
    total = 0
    test_loss = 0
    correct = 0
    class_correct = np.zeros(args.num_classes)
    class_total = np.zeros(args.num_classes)
    class_pred = np.zeros(args.num_classes)

    with torch.no_grad():
        for images, labels in test_loader:
            if torch.cuda.is_available:
                images = images.cuda()
                labels = labels.cuda()

            logits = model(images)
            loss = loss_func(logits, labels)
            test_loss += loss.item()

            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum()

            preds = pred.cpu()
            labels = labels.cpu()
            for i in range(labels.size(0)):
                class_total[labels[i]] += 1
                class_pred[preds[i]] += 1
                if(preds[i] == labels[i]):
                    class_correct[labels[i]] += 1

    loss = 100 * (test_loss / total)
    acc = 100 * float(correct) / float(total)
    std = np.std(100 * class_correct / class_total)
    acc_category = np.around(100 * class_correct / class_total, decimals=2)
    precision_category = np.around(100 * class_correct / class_pred, decimals=2)
    print(getTime(), 'Val Loss: {:.2f}, Acc: {:.2f}, Std: {:.2f}'.format(loss, acc, std))

    return loss, acc, std, acc_category, precision_category


def combinateModels(modelList, model_best_scores, modelsIndexs, dataset):
    labels = []
    data = []
    label_sizes = []
    imagePaths, noise_labels = dataset.getData()
    for j in set(modelsIndexs):
        alist = np.argwhere(modelsIndexs == j)
        print("Load " + modelList[int(j)] + ", label classes: " + str(alist.squeeze().tolist()))
        model = createModel(args.pretrain)
        model.load_state_dict(torch.load(modelList[int(j)]))

        for i in alist:
            labels_index = np.argwhere(noise_labels == i).squeeze()
            get_data = np.take(imagePaths, labels_index).squeeze()
            get_labels = np.take(noise_labels, labels_index).squeeze()
            pred_data, pred_labels, pred_rates = predictByTarget(get_data, get_labels, model, i)
            
            data.extend(pred_data.tolist())
            labels.extend(pred_labels.tolist())
            label_sizes.append(len(pred_labels))

    print('combinate label_sizes', label_sizes)
    return np.array(data), np.array(labels)


def predictByTarget(get_data, get_labels, model, target):
    model.eval()
    preds = []
    rates = []
    # Prepare new data loader by class
    transform_test = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
    new_dataset = Clothing1M_Dataset(get_data, get_labels, args.data_root, transform_test)
    new_dataset_loader = DataLoader(dataset=new_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    with torch.no_grad():
        for images, labels in new_dataset_loader:
            if torch.cuda.is_available:
                images = images.cuda()
                labels = labels.cuda()

            logits = model(images)
            outputs = F.softmax(logits, dim=1)
            rate, pred = torch.max(outputs.data, 1)
            preds.append(pred)
            rates.append(rate)

    preds = torch.cat(preds, dim=0).cpu().numpy()
    rates = torch.cat(rates, dim=0).cpu().numpy()
    labels_index = np.argwhere(preds == target).squeeze()
    data = np.take(get_data, labels_index).squeeze()
    preds = np.take(preds, labels_index).squeeze()
    rates = np.take(rates, labels_index).squeeze()

    return data, preds, rates


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_save_dir = args.model_dir
    if not os.path.exists(model_save_dir):
        os.system('mkdir -p %s' % (model_save_dir))
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(256, padding=32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load data index file
    kvDic = np.load(args.data_root + 'Clothing1m-data.npy', allow_pickle=True).item()
    
    # Prepare train data loader
    original_train_data = kvDic['train_data']
    original_train_labels = kvDic['train_labels']
    shuffle_index = np.arange(len(original_train_labels), dtype=int)
    np.random.shuffle(shuffle_index)
    train_data = original_train_data[shuffle_index]
    train_labels = original_train_labels[shuffle_index]
    train_dataset = Clothing1M_Dataset(train_data, train_labels, args.data_root, transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    predict_dataset = Clothing1M_Dataset(train_data, train_labels, args.data_root, transform_test)

    val_data = kvDic['clean_val_data']
    val_labels = kvDic['clean_val_labels']
    val_dataset = Clothing1M_Dataset(val_data, val_labels, args.data_root, transform_test)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)
    
    test_data = kvDic['test_data']
    test_labels = kvDic['test_labels']
    test_dataset = Clothing1M_Dataset(test_data, test_labels, args.data_root, transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)
    
    # Loss function
    train_nums = np.zeros(args.num_classes, dtype=int)
    val_nums = np.zeros(args.num_classes, dtype=int)
    for item in val_labels:
        val_nums[item] += 1
    for item in train_labels:
        train_nums[item] += 1
    class_weights = torch.FloatTensor(np.mean(train_nums) / train_nums * val_nums / np.mean(val_nums)).cuda()
    ceriation = nn.CrossEntropyLoss(weight=class_weights).cuda()

    best_val_acc = 0
    best_test_acc = 0
    best_model_name = ""
    for outer_loop in range(args.max_outer_loop):
        model = createModel(args.pretrain)
        modelList = np.array([""])
        model_best_scores = np.zeros(args.num_classes, dtype=float)
        model_indexs = np.zeros(args.num_classes, dtype=int)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
        scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
        
        for inner_loop in range(args.max_inner_loop):
            print(getTime(), "Outer", outer_loop, "Inner", inner_loop, "begin...")
            for epoch in range(args.n_epoch):
                train(model, train_loader, optimizer, ceriation, epoch)
                val_loss, val_acc, val_std, val_class_acc, val_class_precision = evaluate_val(val_loader, model, ceriation)
                scheduler.step()
            
                model_scores = args.beta * val_class_acc + (1 - args.beta) * val_class_precision
                filepath = model_save_dir + "/" + str(outer_loop) + "-" + str(inner_loop) + "-" + str(epoch) + "-" + str(round(val_acc, 2)) + ".hdf5"
                for i in range(args.num_classes):
                    if(model_scores[i] > model_best_scores[i]):
                        model_best_scores[i] = model_scores[i]
                        model_indexs[i] = len(modelList)

                if(val_acc > best_val_acc):
                    test_acc, _ = evaluate(model, test_loader, ceriation, "Epoch " + str(epoch) + " Test Acc:")
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_model_name = filepath

                # save model
                modelList = np.append(modelList, filepath)
                torch.save(model.state_dict(), filepath)

            # update train dataset
            # print(getTime(), "Model_best_scores", np.around(model_best_scores, decimals=2), np.around(np.average(model_best_scores), decimals=2), "model indexs", model_indexs, "modelList", modelList)
            train_data, train_labels = combinateModels(modelList, model_best_scores, model_indexs, predict_dataset)
            train_dataset = Clothing1M_Dataset(train_data, train_labels, args.data_root, transform)
            train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
            
            # update loss function
            train_nums = np.zeros(args.num_classes, dtype=int)
            for item in train_labels:
                train_nums[int(item)] += 1
            class_weights = torch.FloatTensor(np.mean(train_nums) / train_nums * val_nums / np.mean(val_nums)).cuda()
            ceriation = nn.CrossEntropyLoss(weight=class_weights).cuda()

    print("Best_test_accuracy:", best_test_acc, ", best_model_name:", best_model_name)


if __name__ == '__main__':
    main()
