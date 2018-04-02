def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def import_data():
    import numpy as np
    dic = unpickle('../doc/data_batch_1')
    data = np.array(dic['data'])
    labels = np.array(dic['labels'])

    for i in range(2, 6):
        dic = unpickle('../doc/data_batch_' + str(i))
        data = np.concatenate((data, np.array(dic['data'])), axis=0)
        labels = np.concatenate((labels, np.array(dic['labels'])), axis=0)

    dic_test = unpickle('../doc/test_batch')
    data_test = np.array(dic_test['data'])
    labels_test = np.array(dic_test['labels'])
    return data, labels, data_test, labels_test
