

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Place dataset at ./data/employee_attrition.csv with 'Attrition' column (1=yes,0=no)
def load_data():
    import os
    path = './data/employee_attrition.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        # synthetic small dataset
        return pd.DataFrame({
            'Age':[29,35,50,23,40,30],
            'MonthlyIncome':[3000,6000,15000,2200,8000,4000],
            'YearsAtCompany':[1,5,20,0,10,2],
            'Attrition':[1,0,0,1,0,1]
        })

def main():
    df = load_data()
    print('Preview:', df.head())
    if 'Attrition' not in df.columns:
        raise ValueError('Dataset must contain Attrition column')
    y = df['Attrition']
    X = df.drop(columns=['Attrition'])
    # simple model
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    # Interpretability: feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print('Feature importances:\n', importances)
    # Business impact: compute how many employees predicted to leave and potential cost (example assumptions)
    predicted_leave = (model.predict(X) == 1).sum()
    avg_replacement_cost = 20000  # example
    print(f'Predicted leavers: {predicted_leave}, estimated replacement cost: {predicted_leave * avg_replacement_cost}')

if __name__ == '__main__':
    main()
