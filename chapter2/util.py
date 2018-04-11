def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_data():
    import numpy as np
    data_train = np.load("../doc/data_sum.npy")
    data_test = np.load("../doc/data_test.npy")

    dic = unpickle('../doc/data_batch_1')
    labels_train = np.array(dic[b'labels'])
    for i in range(2, 6):
        dic = unpickle('../doc/data_batch_' + str(i))
        labels_train = np.concatenate((labels_train, np.array(dic[b'labels'])), axis=0)
    dic_test = unpickle('../doc/test_batch')
    labels_test = np.array(dic_test[b'labels'])

    # get the sample
    num_training = 5000
    mask = list(range(num_training))
    data_train = data_train[mask]
    labels_train = labels_train[mask]

    num_test = 500
    mask = list(range(num_test))
    data_test = data_test[mask]
    labels_test = labels_test[mask]
    return data_train, data_test, labels_train, labels_test


def show_img(x, labels):
    import numpy as np
    import matplotlib.pyplot as plt
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(labels == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(x[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()
