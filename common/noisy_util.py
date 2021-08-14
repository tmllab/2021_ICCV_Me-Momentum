import torch
import torch.nn.functional as F

import numpy as np
from math import inf
from scipy import stats
from numpy.testing import assert_array_almost_equal


def dataset_split(train_images, train_labels, noise_rate=0.5, noise_type='symmetric', split_per=0.9, random_seed=1, num_classes=10, include_noise=False):

    if include_noise:
        noise_rate = noise_rate * (1 - 1 / num_classes)
        print("include_noise True, new real nosie rate:", noise_rate)

    clean_train_labels = train_labels[:, np.newaxis]
    if(noise_type == 'pairflip'):
        noisy_labels, real_noise_rate, transition_matrix = noisify_pairflip(clean_train_labels, noise=noise_rate, random_state=random_seed, nb_classes=num_classes)
    elif(noise_type == 'instance'):
        norm_std = 0.1
        if(len(train_images.shape) == 2):
            feature_size = train_images.shape[1]
        else:
            feature_size = 1
            for i in range(1, len(train_images.shape)):
                feature_size = int(feature_size * train_images.shape[i])

        if torch.is_tensor(train_images) is False:
            data = torch.from_numpy(train_images)
        else:
            data = train_images

        data = data.type(torch.FloatTensor)
        targets = torch.from_numpy(train_labels)
        dataset = zip(data, targets)
        noisy_labels = get_instance_noisy_label(noise_rate, dataset, targets, num_classes, feature_size, norm_std, random_seed)
    elif(noise_type == 'oneflip'):
        noisy_labels, real_noise_rate, transition_matrix = noisify_oneflip(clean_train_labels, noise=noise_rate, random_state=random_seed, nb_classes=num_classes)
    else:
        noisy_labels, real_noise_rate, transition_matrix = noisify_multiclass_symmetric(clean_train_labels, noise=noise_rate, random_state=random_seed, nb_classes=num_classes)

    clean_train_labels = clean_train_labels.squeeze()
    noisy_labels = noisy_labels.squeeze()
    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples * split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]
    train_clean_labels, val_clean_labels = clean_train_labels[train_set_index], clean_train_labels[val_set_index]

    return train_set, val_set, train_labels, val_labels, train_clean_labels, val_clean_labels


def get_instance_noisy_label(n, newdataset, labels, num_classes, feature_size, norm_std, seed):
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    if torch.cuda.is_available():
        labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)
    if torch.cuda.is_available():
        W = torch.FloatTensor(W).cuda()
    else:
        W = torch.FloatTensor(W)
    for i, (x, y) in enumerate(newdataset):
        if torch.cuda.is_available():
            x = x.cuda()
            x = x.reshape(feature_size)

        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l1 = [i for i in range(label_num)]
    new_label = [np.random.choice(l1, p=P[i]) for i in range(labels.shape[0])]
    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1

    return np.array(new_label)


def noisify_oneflip(y_train, noise, random_state=1, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[1, 1], P[1, 2] = 1. - n, n
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    return y_train, actual_noise, P


def noisify_pairflip(y_train, noise, random_state=1, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    return y_train, actual_noise, P


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1. - n
        P[nb_classes - 1, nb_classes - 1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    return y_train, actual_noise, P


def multiclass_noisify(y, P, random_state=1):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # i is np.array, such as [1]
        if not isinstance(i, np.ndarray):
            i = [i]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=1):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=1, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=1, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate


def get_transition_matrix(dataset, noise_type, noise_rate):
    if dataset == "CIFAR10" or dataset == "cifar10":
        nb_classes = 10
    elif dataset == "CIFAR100" or dataset == "cifar100":
        nb_classes = 100

    if noise_type == "symmetric":
        transition_matrix = np.ones((nb_classes, nb_classes))
        transition_matrix = (noise_rate / (nb_classes - 1)) * transition_matrix
        if noise_rate > 0.0:
            transition_matrix[0, 0] = 1. - noise_rate
            for i in range(1, nb_classes - 1):
                transition_matrix[i, i] = 1. - noise_rate
                transition_matrix[nb_classes - 1, nb_classes - 1] = 1. - noise_rate

    elif noise_type == "pairflip":
        transition_matrix = np.eye(nb_classes)
        if noise_rate > 0.0:
            transition_matrix[0, 0], transition_matrix[0, 1] = 1. - noise_rate, noise_rate
            for i in range(1, nb_classes - 1):
                transition_matrix[i, i], transition_matrix[i, i + 1] = 1. - noise_rate, noise_rate
            transition_matrix[nb_classes - 1, nb_classes - 1], transition_matrix[nb_classes - 1, 0] = 1. - noise_rate, noise_rate

    return transition_matrix
