import os
import os.path
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms

from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from common.tools import createModel, evaluate_val, evaluate, predict, train, isSame, getTime, Train_Dataset, write_logs
from common.noisy_util import dataset_split
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
print("PyTorch version:", torch.__version__)
os.system('nvidia-smi')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, help='seed number', default=1)
parser.add_argument('--lr', type=float, help='initial learning rate', default=0.01)
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--model_dir', type=str, help='dir to save model files', default='model')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='mnist')
parser.add_argument('--data_percent', default=0.9, type=float, help='data number percent')
parser.add_argument('--data_path', type=str, default='./data', help='data directory')
parser.add_argument('--noise_type', type=str, help='pairflip, symmetric, instance', default='symmetric')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
parser.add_argument('--max_outer_loop', type=int, help='max outer loop', default=3)
parser.add_argument('--max_inner_loop', type=int, help='max inner loop', default=20)
args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# create the model dir 
model_save_dir = args.model_dir
if not os.path.exists(model_save_dir):
    os.system('mkdir -p %s' % (model_save_dir))

# mnist, cifar10, cifar100
if args.dataset == 'mnist':
    args.num_classes = 10
    args.modelName = 'Lenet'
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
    train_set = MNIST(root=args.data_path, train=True, download=True)
    test_dataset = MNIST(root=args.data_path, train=False, transform=transform_test, download=True)
elif args.dataset == 'cifar10':
    args.num_classes = 10
    args.modelName = 'ResNet18'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = CIFAR10(root=args.data_path, train=True, download=True)
    test_dataset = CIFAR10(root=args.data_path, train=False, transform=transform_test, download=True)
elif args.dataset == 'cifar100':
    args.num_classes = 100
    args.modelName = 'ResNet34'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
    train_set = CIFAR100(root=args.data_path, train=True, download=True)
    test_dataset = CIFAR100(root=args.data_path, train=False, transform=transform_test, download=True)


# if train_clean_labels is not None, the program will check label precision for each class.
def update_confident_examples(cnn, predict_dataset, train_clean_labels):
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, num_workers=8, shuffle=False, pin_memory=True)
    train_data, train_noisy_labels = predict_dataset.getData()
    preds = predict(predict_loader, cnn)
    new_train_data, new_train_labels = isSame(preds, train_noisy_labels, train_clean_labels, train_data, args.num_classes)
    train_dataset = Train_Dataset(new_train_data, new_train_labels, transform_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    return train_loader


def main():
    # Generate noisy labels
    train_data, val_data, train_noisy_labels, val_noisy_labels, _, _ = dataset_split(train_set.data, np.array(train_set.targets), args.noise_rate, args.noise_type, args.data_percent, args.seed, args.num_classes)
    train_dataset = Train_Dataset(train_data, train_noisy_labels, transform_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_dataset = Train_Dataset(val_data, val_noisy_labels, transform_train)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size * 2, num_workers=8, shuffle=False, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size * 2, num_workers=8, shuffle=False, pin_memory=True)
    predict_dataset = Train_Dataset(train_data, train_noisy_labels, transform_train)
    
    best_val_acc = 0
    best_test_acc = 0
    save_file_path_latest = model_save_dir + "/" + args.dataset + "-" + args.noise_type + "-" + str(args.noise_rate) + "-" + str(args.seed) + "_latest.hdf5"
    save_file_path_best = model_save_dir + "/" + args.dataset + "-" + args.noise_type + "-" + str(args.noise_rate) + "-" + str(args.seed) + "_best.hdf5"
    ceriation = nn.CrossEntropyLoss().cuda()
    for outer_loop in range(args.max_outer_loop):
        model = createModel(args.modelName, args.num_classes)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

        n_add = 0        
        inner_val_acc = 0
        last_round_acc = 0
        # early stopping round
        for epoch in range(args.n_epoch):
            train(model, train_loader, optimizer, ceriation, epoch)
            val_acc, _ = evaluate_val(model, val_loader, args.num_classes, "Epoch " + str(epoch) + " Val Acc:")
            # proposed early stopping trick
            if(val_acc > inner_val_acc + n_add):
                inner_val_acc = val_acc
                print(getTime(), "Save Model:" + save_file_path_latest)
                torch.save(model.state_dict(), save_file_path_latest)
                n_add = 0  # increased, reset n_add
            else:
                n_add += 0.1

            if n_add > 2:
                print("No improved in 20 epoch, break!")  # for saving time
                break

        # Normally, the method stops by itself in the inner loop
        for inner_loop in range(args.max_inner_loop - 1):
            print(getTime(), "Load " + save_file_path_latest + ", evaluate it and predict labels...")
            model = createModel(args.modelName, args.num_classes)
            model.load_state_dict(torch.load(save_file_path_latest))
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
            scheduler = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)   
            train_loader = update_confident_examples(model, predict_dataset, None)
            
            for epoch in range(args.n_epoch):
                train(model, train_loader, optimizer, ceriation, epoch)
                val_acc, val_std = evaluate_val(model, val_loader, args.num_classes, "Epoch " + str(epoch) + " Val Acc:")
                scheduler.step()

                if(val_acc > inner_val_acc):
                    inner_val_acc = val_acc
                    print(getTime(), "Save Model:" + save_file_path_latest)
                    torch.save(model.state_dict(), save_file_path_latest)
                    if(val_acc > best_val_acc):
                        best_val_acc = val_acc
                        test_acc, _ = evaluate(model, test_loader, ceriation, "Epoch " + str(epoch) + " Test Acc:")
                        best_test_acc = test_acc
                        torch.save(model.state_dict(), save_file_path_best)
                        print(getTime(), "Save Model:" + save_file_path_best)

            # if the accuracy is non-increasing in the loop, stopping training in the inner loop;
            if(inner_val_acc > last_round_acc):
                last_round_acc = inner_val_acc
            else:
                print("Val accrucy was not improved! Num:" + str(outer_loop + 1))
                break

    print(getTime(), "Test acc", best_test_acc)
    write_logs("Me-Momentum", args, best_test_acc)


if __name__ == '__main__':
    main()
