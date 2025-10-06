

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report

def main():
    X, y = make_classification(n_samples=2000, n_features=30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    # Random Forest Grid
    rf = RandomForestClassifier(random_state=42)
    rf_grid = {'n_estimators':[100,200],'max_depth':[None,10,20]}
    gs_rf = GridSearchCV(rf, rf_grid, cv=3, n_jobs=-1)
    gs_rf.fit(X_train, y_train)
    print('RF best:', gs_rf.best_params_)
    # SVM Randomized
    svc = SVC()
    svc_dist = {'C':[0.1,1,10],'kernel':['rbf','linear']}
    rs_svc = RandomizedSearchCV(svc, svc_dist, n_iter=6, cv=3, n_jobs=-1, random_state=42)
    rs_svc.fit(X_train, y_train)
    print('SVC best:', rs_svc.best_params_)
    # Gradient Boosting Grid
    gb = GradientBoostingClassifier(random_state=42)
    gb_grid = {'n_estimators':[100,200], 'learning_rate':[0.01,0.1]}
    gs_gb = GridSearchCV(gb, gb_grid, cv=3, n_jobs=-1)
    gs_gb.fit(X_train, y_train)
    print('GB best:', gs_gb.best_params_)

if __name__ == '__main__':
    main()
