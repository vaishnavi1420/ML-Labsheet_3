

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def main():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'L1': LogisticRegression(penalty='l1', solver='saga', max_iter=5000, C=1.0),
        'L2': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000, C=1.0)
    }
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        print('\n---', name, '---')
        print('Accuracy:', accuracy_score(y_test, preds))
        # sparsity
        coef = model.coef_.ravel()
        nonzero = np.sum(coef != 0)
        print('Number of non-zero coefficients:', int(nonzero), '/', coef.size)
        print(classification_report(y_test, preds))

if __name__ == '__main__':
    main()
