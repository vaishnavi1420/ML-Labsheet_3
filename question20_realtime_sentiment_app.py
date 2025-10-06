
import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Simple streamlit app to classify sentiment using a pretrained model or train on small data
def train_sample_model():
    # tiny sample training for demo only
    data = pd.DataFrame({
        'text':['I love this product','This is terrible','Absolutely fantastic','Worst ever','Good value'],
        'label':[1,0,1,0,1]
    })
    vec = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vec.fit_transform(data['text'])
    model = LogisticRegression()
    model.fit(X, data['label'])
    joblib.dump((vec, model), 'sentiment_model.joblib')
    return vec, model

def main():
    st.title('Real-time Sentiment Classifier (Demo)')
    st.write('Enter a review to classify positive (1) or negative (0).')
    if st.button('Train demo model'):
        vec, model = train_sample_model()
        st.success('Demo model trained and saved to sentiment_model.joblib')
    uploaded = st.text_area('Type review here')
    if st.button('Classify'):
        try:
            vec, model = joblib.load('sentiment_model.joblib')
        except Exception:
            vec, model = train_sample_model()
        X = vec.transform([uploaded])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
        st.write('Prediction:', int(pred))
        if proba is not None:
            st.write('Probability:', proba)

if __name__ == '__main__':
    main()
