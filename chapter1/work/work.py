def lr_work(data_train, data_test, data_train_lbp, data_test_lbp, data_train_hog, data_test_hog, labels_train, labels_test):
    from sklearn.linear_model import LogisticRegression

    # lr = LogisticRegression(C=1e5, multi_class='multinomial', penalty='l2', solver='lbfgs', tol=0.1)
    # lr.fit(data_train, labels_train)
    # print('The accuracy of lbfgs LogisticRegression is', lr.score(data_test, labels_test))
    # lr = LogisticRegression(C=1e5, multi_class='multinomial', penalty='l2', solver='lbfgs', tol=0.1)
    # lr.fit(data_train_lbp, labels_train)
    # print('The accuracy of lbfgs(lbp) LogisticRegression is', lr.score(data_test_lbp, labels_test))
    # lr = LogisticRegression(C=1e5, multi_class='multinomial', penalty='l2', solver='lbfgs', tol=0.1)
    # lr.fit(data_train_hog, labels_train)
    # print('The accuracy of lbfgs(hog) LogisticRegression is', lr.score(data_test_hog, labels_test))

    lr = LogisticRegression(C=1e5, multi_class='multinomial', penalty='l2', solver='sag', tol=0.1)
    lr.fit(data_train, labels_train)
    print('The accuracy of sag LogisticRegression is', lr.score(data_test, labels_test))
    # lr = LogisticRegression(C=1e5, multi_class='multinomial', penalty='l2', solver='sag', tol=0.1)
    # lr.fit(data_train_lbp, labels_train)
    # print('The accuracy of sag(lbp) LogisticRegression is', lr.score(data_test_lbp, labels_test))
    lr = LogisticRegression(C=1e5, multi_class='multinomial', penalty='l2', solver='sag', tol=0.1)
    lr.fit(data_train_hog, labels_train)
    print('The accuracy of sag(hog) LogisticRegression is', lr.score(data_test_hog, labels_test))

    # lr = LogisticRegression(C=1e5, multi_class='multinomial', penalty='l2', solver='newton-cg', tol=0.1)
    # lr.fit(data_train, labels_train)
    # print('The accuracy of newton-cg LogisticRegression is', lr.score(data_test, labels_test))
    # lr = LogisticRegression(C=1e5, multi_class='multinomial', penalty='l2', solver='newton-cg', tol=0.1)
    # lr.fit(data_train_lbp, labels_train)
    # print('The accuracy of newton-cg(lbp) LogisticRegression is', lr.score(data_test_lbp, labels_test))
    # lr = LogisticRegression(C=1e5, multi_class='multinomial', penalty='l2', solver='newton-cg', tol=0.1)
    # lr.fit(data_train_hog, labels_train)
    # print('The accuracy of newton-cg(hog) LogisticRegression is', lr.score(data_test_hog, labels_test))


def knn_work(data_train_lbp, data_test_lbp, data_train_hog, data_test_hog, labels_train, labels_test):
    n = 7

    from sklearn import neighbors
    knn_hog = neighbors.KNeighborsClassifier(n, weights='uniform')
    knn_hog.fit(data_train_hog, labels_train)
    print('KNN where n=5 and input are hogs: ', knn_hog.score(data_test_hog, labels_test))

    knn_lbp = neighbors.KNeighborsClassifier(n, weights='uniform')
    knn_lbp.fit(data_train_lbp, labels_train)
    print('KNN where n=5 and input are lbps: ', knn_lbp.score(data_test_lbp, labels_test))


def svm_work(data_train, data_test, data_train_lbp, data_test_lbp, data_train_hog, data_test_hog, labels_train, labels_test):
    from sklearn import svm

    # svc = svm.SVC(C=1e3, kernel='rbf', gamma=0.6)
    # svc.fit(data_train, labels_train)
    # print('The accuracy of rbf SVM is', svc.score(data_test, labels_test))
    # svc = svm.SVC(C=1e3, kernel='rbf', gamma=0.6)
    # svc.fit(data_train_lbp, labels_train)
    # print('The accuracy of rbf SVM(lbp) is', svc.score(data_test_lbp, labels_test))
    # svc = svm.SVC(C=1e3, kernel='rbf', gamma=0.6)
    # svc.fit(data_train_hog, labels_train)
    # print('The accuracy of rbf SVM(hog) is', svc.score(data_test_hog, labels_test))

    # svc = svm.SVC(C=1e3, kernel='poly')
    # svc.fit(data_train, labels_train)
    # print('The accuracy of poly SVM is', svc.score(data_test, labels_test))
    # svc = svm.SVC(C=1e3, kernel='poly')
    # svc.fit(data_train_lbp, labels_train)
    # print('The accuracy of poly SVM(lbp) is', svc.score(data_test_lbp, labels_test))
    # svc = svm.SVC(C=1e3, kernel='poly')
    # svc.fit(data_train_hog, labels_train)
    # print('The accuracy of poly SVM(hog) is', svc.score(data_test_hog, labels_test))

    # svc = svm.SVC(C=1e3, kernel='sigmoid', gamma=0.6)
    # svc.fit(data_train, labels_train)
    # print('The accuracy of sigmoid SVM is', svc.score(data_test, labels_test))
    # svc = svm.SVC(C=1e3, kernel='sigmoid', gamma=0.6)
    # svc.fit(data_train_lbp, labels_train)
    # print('The accuracy of sigmoid SVM(lbp) is', svc.score(data_test_lbp, labels_test))
    # svc = svm.SVC(C=1e3, kernel='sigmoid', gamma=0.6)
    # svc.fit(data_train_hog, labels_train)
    # print('The accuracy of sigmoid SVM(hog) is', svc.score(data_test_hog, labels_test))

    svc = svm.SVC(C=1e3, kernel='linear')
    svc.fit(data_train, labels_train)
    print('The accuracy of linear SVM is', svc.score(data_test, labels_test))
    svc = svm.SVC(C=1e3, kernel='linear')
    svc.fit(data_train_lbp, labels_train)
    print('The accuracy of linear SVM(lbp) is', svc.score(data_test_lbp, labels_test))
    svc = svm.SVC(C=1e3, kernel='linear')
    svc.fit(data_train_hog, labels_train)
    print('The accuracy of linear SVM(hog) is', svc.score(data_test_hog, labels_test))
