

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print('Top features:\n', importances.head(10))
    plt.figure(figsize=(8,6))
    importances.head(20).plot(kind='bar')
    plt.title('Top 20 feature importances')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
