

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, accuracy_score

def main():
    X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
    gs = GridSearchCV(ada, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)
    print('Best params:', gs.best_params_)
    best = gs.best_estimator_
    preds = best.predict(X_test)
    print('Test accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

if __name__ == '__main__':
    main()
