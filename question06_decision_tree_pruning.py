

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def main():
    data = load_diabetes()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    # Train fully grown tree
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    print('Train RMSE:', mean_squared_error(y_train, dt.predict(X_train), squared=False))
    print('Test RMSE:', mean_squared_error(y_test, dt.predict(X_test), squared=False))

    # Cost complexity pruning path
    path = dt.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    clfs = []
    for a in ccp_alphas:
        clf = DecisionTreeRegressor(random_state=42, ccp_alpha=a)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    train_scores = [mean_squared_error(y_train, c.predict(X_train), squared=False) for c in clfs]
    test_scores = [mean_squared_error(y_test, c.predict(X_test), squared=False) for c in clfs]

    plt.figure(figsize=(8,5))
    plt.plot(ccp_alphas, train_scores, marker='o', label='train RMSE')
    plt.plot(ccp_alphas, test_scores, marker='o', label='test RMSE')
    plt.xlabel('ccp_alpha')
    plt.ylabel('RMSE')
    plt.title('Cost-Complexity Pruning')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # Depth control pruning
    depths = range(1, 21)
    train_rmse = []
    test_rmse = []
    for d in depths:
        clf = DecisionTreeRegressor(random_state=42, max_depth=d)
        clf.fit(X_train, y_train)
        train_rmse.append(mean_squared_error(y_train, clf.predict(X_train), squared=False))
        test_rmse.append(mean_squared_error(y_test, clf.predict(X_test), squared=False))
    plt.figure(figsize=(8,5))
    plt.plot(depths, train_rmse, marker='o', label='train RMSE')
    plt.plot(depths, test_rmse, marker='o', label='test RMSE')
    plt.xlabel('max_depth')
    plt.ylabel('RMSE')
    plt.title('Depth-based pruning')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
