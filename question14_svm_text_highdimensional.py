

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Place dataset at ./data/spam.csv with columns 'text' and 'label'
def load_data():
    import os
    path = './data/spam.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        # small dummy
        return pd.DataFrame({'text':['Free lottery win','Hi how are you','Limited offer now','Meeting at 10am'],
                             'label':[1,0,1,0]})

def main():
    df = load_data()
    X = df['text']
    y = df['label']
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=20000, stop_words='english')
    X_t = vec.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42, stratify=y)
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    preds = svc.predict(X_test)
    print(classification_report(y_test, preds))
    print('\nAdvantages of SVM in high-dim text: margin maximization, works well with sparse data, effective with linear kernels and TF-IDF.')

if __name__ == '__main__':
    main()
