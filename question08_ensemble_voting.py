

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score,classification_report

def main():
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    lr = LogisticRegression(max_iter=2000)
    dt = DecisionTreeClassifier(max_depth=5)
    svc = SVC(probability=True)
    ensemble = VotingClassifier(estimators=[('lr',lr),('dt',dt),('svc',svc)], voting='soft')
    for name, model in [('Logistic',lr),('DecisionTree',dt),('SVC',svc),('Ensemble',ensemble)]:
        model.fit(X_train,y_train)
        preds = model.predict(X_test)
        print('\n==', name, 'Accuracy:', accuracy_score(y_test,preds))
        print(classification_report(y_test,preds))

if __name__ == '__main__':
    main()
