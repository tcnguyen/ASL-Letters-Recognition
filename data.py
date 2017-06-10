# We use only color image (not depth image)
# dataset object has the same structure as the dataset5 folder

import glob
import numpy as np
from PIL import Image

PATH_DATASET = './datasets/dataset5/'
PATH_DATASET_120 = './datasets/dataset5_120/'
PATH_DATASET_64 = './datasets/dataset5_64/'

people = "ABCDE"
letters = "abcdefghiklmnopqrstuvwxy"
letter_indices = dict((key, value) for (key, value) in zip(letters, range(len(letters))))


def get_dataset(type="original"):
    dataset = {}

    if type == "128":
        path = PATH_DATASET_120
    elif type == "64":
        path = PATH_DATASET_64
    else:
        path = PATH_DATASET


    for p in people:
        dataset[p] = {}
        for c in letters:
            dataset[p][c] = glob.glob(path + p + '/' + c + '/color_*.png')

    # total number of images
    total_images = sum([len(dataset[p][c]) for p in people for c in letters])
    print("Total number of images in dataset5: ", total_images)

    return dataset


def split_train_validation_test(dataset):
    # Keep 5% of all the data for testing ==> 3200 images for testing
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for p in people:
        for c in letters:
            data = dataset[p][c]
            y = letter_indices[c]

            n = len(data)
            n_train = int(n*0.925)

            train_data.extend(data[:n_train])
            train_label.extend([y for _ in range(n_train)])

            test_data.extend(data[n_train:])

            test_label.extend([y for _ in range(n - n_train)])

    # randomize data orders
    indices = np.arange(len(train_data))
    np.random.shuffle(indices)

    train_data = np.array(train_data)[indices]
    train_label = np.array(train_label)[indices]

    # one_hot_encoding
    train_label_onehot = np.zeros((len(train_label), 24))
    train_label_onehot[np.arange(len(train_label)), train_label] = 1

    n_validation = 10000

    # keep last 1000 for validation
    validation_data = train_data[0:n_validation]
    validation_label_onehot = train_label_onehot[0:n_validation]

    train_data = train_data[n_validation:]
    train_label = train_label[n_validation:]
    train_label_onehot = train_label_onehot[n_validation:]

    # one_hot_encoding
    test_label_onehot = np.zeros((len(test_label), 24))
    test_label_onehot[np.arange(len(test_label)), test_label] = 1

    print("Train data size: ", len(train_data))
    print("Train label size: ", len(train_label_onehot))
    print("Validation data size: ", len(validation_data))
    print("Validation label size: ", len(validation_label_onehot))

    print("Test data size: ", len(test_data))
    print("Test label size: ", len(test_label_onehot))

    return train_data, train_label_onehot, validation_data, validation_label_onehot, test_data, test_label_onehot

def get_data_shapes(data):
    # Get images shape data
    data_shapes = np.empty(len(data), dtype=object)
    count = 0
    for i in data:
        data_shapes[count] = Image.open(i).size
        count += 1

    data_widths = [k[0] for k in data_shapes]
    data_heights = [k[1] for k in data_shapes]

    print("min image width = ", min(data_widths))
    print("max image width = ", max(data_widths))
    print("avg image width = ", sum(data_widths) / len(data_widths))

    print("")
    print("min image height = ", min(data_heights))
    print("max image height = ", max(data_heights))
    print("avg image height = ", sum(data_heights) / len(data_heights))

    #plt.scatter(data_widths, data_heights)
