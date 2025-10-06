
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Place dataset at ./data/fake_news.csv with columns: 'text', 'label' (1=fake,0=real)
def load_data():
    import os
    path = './data/fake_news.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        # create a tiny dummy dataset
        data = {'text': ['The sky is blue','Aliens landed today','Government confirms cure','Clickbait headline'],
                'label': [0,1,0,1]}
        return pd.DataFrame(data)

def main():
    df = load_data()
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    vec = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
    X_train_t = vec.fit_transform(X_train)
    X_test_t = vec.transform(X_test)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(max_depth=10),
        'LinearSVM': LinearSVC(max_iter=2000)
    }

    for name, model in models.items():
        model.fit(X_train_t, y_train)
        preds = model.predict(X_test_t)
        print('\n====', name, '====')
        print('Accuracy:', accuracy_score(y_test, preds))
        print(classification_report(y_test, preds))

if __name__ == '__main__':
    main()
