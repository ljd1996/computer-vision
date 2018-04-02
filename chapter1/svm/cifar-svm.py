from chapter1.util.util import import_data
from sklearn import svm


if __name__ == '__main__':
    data, labels, data_test, labels_test = import_data()
    svc = svm.SVC()
    svc.fit(data[:5000], labels[:5000])
    score = svc.score(data_test, labels_test)
    print('The accuracy of SVM is', score)
