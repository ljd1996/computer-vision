from chapter1.util.util import import_data

if __name__ == '__main__':
    data, labels, data_test, labels_test = import_data()

    from sklearn import linear_model
    lr = linear_model.LogisticRegression(solver='sag', multi_class='multinomial', C=1)
    lr.fit(data, labels)
    score = lr.score(data_test, labels_test)
    print('The accuracy of LogisticRegression is', score)
