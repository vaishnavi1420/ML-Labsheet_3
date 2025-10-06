
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    base_learners = {
        'DecisionTree': DecisionTreeClassifier(max_depth=4),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic': LogisticRegression(max_iter=2000)
    }
    for name, base in base_learners.items():
        bag = BaggingClassifier(base_estimator=base, n_estimators=50, random_state=42)
        ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        for model_name, model in [('Bagging',bag),('AdaBoost',ada),('GradientBoosting',gb)]:
            model.fit(X_train,y_train)
            preds = model.predict(X_test)
            print(f'\nBase: {name} | Model: {model_name} | Acc: {accuracy_score(y_test,preds)}')
            print(classification_report(y_test,preds))

if __name__ == '__main__':
    main()
