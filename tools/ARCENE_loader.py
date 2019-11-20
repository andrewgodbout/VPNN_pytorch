import torch
import numpy
import torch.utils.data
""" ARCENE_loader
download at https://archive.ics.uci.edu/ml/datasets/Arcene
intended to load and return a custom pytorch dataset containing the ARCENE
dataset.

ARCENE: 100 instances in train or test x 10k attributes x 32 bit x 2 datasets
= 64 MB of memory usage
Will load directly into torch tensors in memory
expects '/ARCENE/arcene_valid.labels', '/ARCENE/arcene_valid.data',
'/ARCENE/arcene_train.labels', '/ARCENE/arcene_train.data'
in folder DATA_PATH
"""


class Arcene_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, preprocessing=None):
        self.data = data
        self.preprocessing = preprocessing
        self.labels = labels

    def __getitem__(self, idx):
        inp = self.data[idx]

        if self.preprocessing is not None:
            inp = self.preprocessing(inp)

        return (inp, self.labels[idx])

    def __len__(self):
        return len(self.data)


def arcene_load(DATA_PATH, preprocessing):
    f = open(DATA_PATH+'/ARCENE/arcene_train.data')
    train_data = f.readlines()  # full text
    train_data = [x.replace(' \n', '') for x in train_data]
    train_data = [x.split(' ') for x in train_data]
    train_data = numpy.array(train_data, dtype=numpy.float32)
    train_data = torch.tensor(train_data)
    f.close()

    f = open(DATA_PATH+'/ARCENE/arcene_train.labels')
    train_labels = f.readlines()
    train_labels = [x.replace(' \n', '') for x in train_labels]
    train_labels = numpy.array(train_labels, dtype=numpy.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    train_labels[train_labels == -1] = 0
    f.close()

    f = open(DATA_PATH+'/ARCENE/arcene_valid.data')
    valid_data = f.readlines()  # full text
    valid_data = [x.replace(' \n', '') for x in valid_data]
    valid_data = [x.split(' ') for x in valid_data]
    valid_data = numpy.array(valid_data, dtype=numpy.float32)
    valid_data = torch.tensor(valid_data)
    f.close()

    f = open(DATA_PATH+'/ARCENE/arcene_valid.labels')
    valid_labels = f.readlines()
    valid_labels = [x.replace(' \n', '') for x in valid_labels]
    valid_labels = numpy.array(valid_labels, dtype=numpy.float32)
    valid_labels = torch.tensor(valid_labels, dtype=torch.long)
    valid_labels[valid_labels == -1] = 0
    f.close()

    train_ds = Arcene_Dataset(train_data, train_labels, preprocessing=preprocessing)
    valid_ds = Arcene_Dataset(valid_data, valid_labels, preprocessing=preprocessing)
    return train_ds, valid_ds
