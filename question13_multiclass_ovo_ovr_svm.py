
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    ovo = OneVsOneClassifier(SVC())
    ovr = OneVsRestClassifier(SVC())
    ovo.fit(X_train, y_train)
    ovr.fit(X_train, y_train)
    for name, model in [('OVO', ovo), ('OVR', ovr)]:
        preds = model.predict(X_test)
        print('\n==', name)
        print(classification_report(y_test, preds))
        cm = confusion_matrix(y_test, preds)
        print('Confusion matrix:\n', cm)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(name + ' Confusion Matrix')
        plt.show()

if __name__ == '__main__':
    main()
