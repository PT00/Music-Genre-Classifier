import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

import pandas as pd

def show_first_ten_rows(file_path='../../data/processed/music_features.csv'):
    df = pd.read_csv(file_path)
    print(df.columns)
    print(df.head(10))


def random_forest_classify(df: pd.DataFrame, category:str):
 
    feature_cols = df.select_dtypes(include=[np.number]).columns
    X = df[feature_cols]
    y = df[category]

    # dzielimy z zachowaniem proporcji
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            random_state=42, 
            test_size=0.2,
            stratify=y
        )


    # można się pobawić parametrami
    rf = RandomForestClassifier(
            n_estimators=1000,
            random_state=42,
            min_samples_split=2,
            n_jobs=-1,
            criterion='gini'
        )

    rf.fit(X_train, y_train)
        
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')


    print(classification_report(y_test, y_pred))
    
    # Print results
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    })
    print("\nTop 10 Most Important Features:")
    print(feature_importance.sort_values('importance', ascending=False).head(10))
    
    return accuracy, y_prob

if __name__ == "__main__":
    # show_first_ten_rows()
    df = pd.read_csv('../../data/processed/music_features.csv')
    random_forest_classify(df, "rock")
