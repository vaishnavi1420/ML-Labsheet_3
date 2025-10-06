
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Create synthetic multicollinear dataset
def create_data(n=500, p=10, rho=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    # base signal
    z = rng.normal(size=(n,1))
    X = z @ (rng.normal(size=(1,p))) + rng.normal(scale=0.1, size=(n,p))
    # true coefficients
    coef = np.array([1.5, -2.0] + [0]*(p-2))
    y = X.dot(coef) + rng.normal(scale=0.5, size=n)
    return X, y

def main():
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    alphas = [0.01, 0.1, 1, 10, 100]
    ridge_coefs = []
    lasso_coefs = []
    for a in alphas:
        r = Ridge(alpha=a).fit(X_train_s, y_train)
        l = Lasso(alpha=a, max_iter=10000).fit(X_train_s, y_train)
        ridge_coefs.append(r.coef_)
        lasso_coefs.append(l.coef_)

    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)

    # coefficient plots
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    for i in range(ridge_coefs.shape[1]):
        axes[0].plot(alphas, ridge_coefs[:,i], marker='o', label=f'coef_{i}')
    axes[0].set_xscale('log')
    axes[0].set_title('Ridge coefficients vs alpha')
    axes[0].set_xlabel('alpha (log scale)')

    for i in range(lasso_coefs.shape[1]):
        axes[1].plot(alphas, lasso_coefs[:,i], marker='o', label=f'coef_{i}')
    axes[1].set_xscale('log')
    axes[1].set_title('Lasso coefficients vs alpha')
    axes[1].set_xlabel('alpha (log scale)')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
