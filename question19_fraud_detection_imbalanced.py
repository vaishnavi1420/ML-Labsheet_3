
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score

def main():
    X, y = make_classification(n_samples=5000, n_features=20, weights=[0.99], flip_y=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    print('Original train class distribution:', np.bincount(y_train))
    # Apply SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print('Resampled distribution:', np.bincount(y_res))
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_res, y_res)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    probs = model.predict_proba(X_test)[:,1]
    print('Average precision (AP):', average_precision_score(y_test, probs))

if __name__ == '__main__':
    main()
