
import numpy as np
from sklearn.datasets import load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def main():
    X, y = make_classification(n_samples=1500, n_features=25, n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    estimators = [
        ('dt', DecisionTreeClassifier(max_depth=5)),
        ('svc', SVC(probability=True))
    ]
    stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=2000), cv=5)
    stack.fit(X_train, y_train)
    preds = stack.predict(X_test)
    print('Stacking acc:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    # Quick comparison with bagging and boosting
    bag = BaggingClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=50).fit(X_train,y_train)
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50).fit(X_train,y_train)
    for name, model in [('Bagging',bag), ('AdaBoost', ada)]:
        p = model.predict(X_test)
        print(f'\n{name} acc:', accuracy_score(y_test,p))
        print(classification_report(y_test,p))

if __name__ == '__main__':
    main()
