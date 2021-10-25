import torchvision
import torchvision.transforms as tf
import numpy as np
from model import Dataset
from torch.utils.data import DataLoader, Subset
def get_dataset(dataset, args):
    if dataset == 'MNIST':
        train_raw_dataset = torchvision.datasets.MNIST(root='../mnist',
                                                       train=True,
                                                       download=False,
                                                       transform=tf.Compose([tf.Resize((32, 32)), tf.ToTensor(),
                                                                             tf.Normalize((0.1307,), (0.3081,))]))

        valid_raw_dataset = torchvision.datasets.MNIST(root='../mnist',
                                                       train=False,
                                                       download=False,
                                                       transform=tf.Compose([tf.Resize((32, 32)), tf.ToTensor(),
                                                                             tf.Normalize((0.1307,), (0.3081,))]))

        # Train and validate only on pictures of args.target_cls
        train_dataset = Dataset(train_raw_dataset, [args.target_cls])
        valid_dataset = Dataset(valid_raw_dataset, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        return train_dataset, valid_dataset
    elif dataset == 'CIFAR':
        trainset = torchvision.datasets.CIFAR10(
            root='../../transformers/data', train=True, download=False,
            transform=tf.Compose([tf.ToTensor()]))
        testset = torchvision.datasets.CIFAR10(
            root='../../transformers/data', train=False, download=False,
            transform=tf.Compose([tf.ToTensor()]))
        allSampleSet = trainset + testset

        classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
        train_inliers = [np.where(np.array(trainset.targets) == class_idx)[0]
                         for class_idx in trainset.class_to_idx.values()]
        train_outliers = [np.where(np.array(trainset.targets) != class_idx)[0]
                          for class_idx in trainset.class_to_idx.values()]
        test_inliers = [np.where(np.array(testset.targets) == class_idx)[0]
                        for class_idx in testset.class_to_idx.values()]
        test_outliers = [np.where(np.array(testset.targets) != class_idx)[0]
                         for class_idx in testset.class_to_idx.values()]

        print(len(train_inliers))

        for i in range(len(classes)):
            test_inliers[i] += len(trainset)
            test_outliers[i] += len(trainset)

            # Drop elements
            train_outliers[i] = np.random.choice(train_outliers[i], size=500, replace=False)
            test_outliers[i] = np.random.choice(test_outliers[i], size=500, replace=False)

        inliers_zip = zip(train_inliers, test_inliers)
        inliers = [np.concatenate((i, j), dtype=np.int64) for i, j in inliers_zip]

        outliers_zip = zip(train_outliers, test_outliers)
        outliers = [np.concatenate((i, j), dtype=np.int64) for i, j in outliers_zip]

        trainloader = [
            DataLoader(
                dataset=Subset(allSampleSet, inds),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=1
            ) for inds in train_inliers]

        testloader = [
            DataLoader(
                dataset=testset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=1
            ) for inds in outliers]
        unified_loaders = list(zip(trainloader, testloader))
        return unified_loaders