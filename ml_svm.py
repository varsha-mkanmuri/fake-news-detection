from sklearn import svm

def svm_train(X_train, y_train):
    rbfsvm = svm.SVC(C=10, probability=True)

    rbfsvm.fit(X_train, y_train)

    return rbfsvm

def svm_predict(X, rbfsvm):

    y_pred = rbfsvm.predict(X)

    return y_pred
