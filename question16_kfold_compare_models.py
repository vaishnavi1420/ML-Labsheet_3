
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def main():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    models = {
        'Logistic': LogisticRegression(max_iter=5000),
        'RandomForest': RandomForestClassifier(n_estimators=200),
        'SVC': SVC()
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        results[name] = scores
        print(name, 'mean acc:', scores.mean(), 'std:', scores.std())
    # visualize
    plt.boxplot([results[m] for m in results.keys()], labels=list(results.keys()))
    plt.title('Model accuracy variation (k-fold)')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == '__main__':
    main()
