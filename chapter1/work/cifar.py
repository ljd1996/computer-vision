from chapter1.util.util import get_lbp_data, get_hog_data, get_data
from chapter1.work.work import lr_work, knn_work, svm_work
import numpy as np


if __name__ == '__main__':
    data_train, data_test, labels_train, labels_test = get_data()
    data_train_lbp, data_test_lbp = get_lbp_data(data_train, data_test)
    data_train_hog, data_test_hog = get_hog_data(data_train, data_test)
    data_train = np.reshape(data_train, (data_train.shape[0], -1))
    data_test = np.reshape(data_test, (data_test.shape[0], -1))

    # lr_work(data_train, data_test, data_train_lbp, data_test_lbp, data_train_hog, data_test_hog, labels_train, labels_test)
    # knn_work(data_train, data_test, data_train_lbp, data_test_lbp, data_train_hog, data_test_hog, labels_train, labels_test)
    svm_work(data_train, data_test, data_train_lbp, data_test_lbp, data_train_hog, data_test_hog, labels_train, labels_test)
