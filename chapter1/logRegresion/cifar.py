def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


if __name__ == '__main__':
    import numpy as np
    dic = unpickle('../doc/data_batch_1')
    data = np.array(dic['data'])
    labels = np.array(dic['labels'])

    dic_test = unpickle('../doc/test_batch')
    data_test = np.array(dic['data'])
    labels_test = np.array(dic['labels'])

    from sklearn import linear_model
    lr = linear_model.LogisticRegression(C=1e5)
    lr.fit(data, labels)
    score = lr.score(data_test, labels_test)
    print(score)
