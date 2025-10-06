

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def main():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Backward elimination via RFE (Recursive Feature Elimination) with logistic regression
    model = LogisticRegression(max_iter=5000)
    selector = RFE(model, n_features_to_select=10, step=1)
    selector = selector.fit(X_train_s, y_train)
    selected_features = X.columns[selector.support_]
    print('RFE selected features:', list(selected_features))
    preds_rfe = model.fit(X_train_s[:, selector.support_], y_train).predict(X_test_s[:, selector.support_])
    print('RFE accuracy:', accuracy_score(y_test, preds_rfe))

    # Lasso feature selection
    lasso = LassoCV(cv=5, max_iter=10000).fit(X_train_s, y_train)
    coef = pd.Series(lasso.coef_, index=X.columns)
    selected_lasso = list(coef[coef!=0].index)
    print('Lasso selected features:', selected_lasso)
    if selected_lasso:
        preds_lasso = LogisticRegression(max_iter=5000).fit(X_train_s[:, coef!=0], y_train).predict(X_test_s[:, coef!=0])
        print('Lasso-based logistic accuracy:', accuracy_score(y_test, preds_lasso))

if __name__ == '__main__':
    main()
