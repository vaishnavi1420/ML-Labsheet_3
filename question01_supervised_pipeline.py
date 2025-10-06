
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Example: use UCI-style CSV placed at ./data/loan_data.csv OR use sklearn's dataset as fallback
def load_data():
    import os
    path = './data/loan_data.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        # fallback: use breast cancer dataset for binary classification
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        df = data.frame
    return df

def main():
    df = load_data()
    print('Data preview:\n', df.head())
    # Basic target detection: if 'target' not in columns, try common names
    if 'target' in df.columns:
        y = df['target']
        X = df.drop(columns=['target'])
    elif 'Class' in df.columns:
        y = df['Class']
        X = df.drop(columns=['Class'])
    else:
        # assume last column is target
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Preprocessing pipelines
    num_pipeline = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    clf = Pipeline([
        ('pre', preprocessor),
        ('model', RandomForestClassifier(random_state=42, n_estimators=100))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(set(y))>1 else None)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print('\nClassification report:\n', classification_report(y_test, preds))
    try:
        probs = clf.predict_proba(X_test)[:,1]
        print('\nROC AUC:', roc_auc_score(y_test, probs))
    except Exception:
        pass

if __name__ == '__main__':
    main()
