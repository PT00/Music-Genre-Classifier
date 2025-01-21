from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def logistic_regression_classifier(df_music: pd.DataFrame, category: str):
    
    df = df_music.drop(columns=["title"])
    df = df[df[category].isin([0, 1])]


    feature_cols = df.select_dtypes(include=[np.number]).columns
    X = df[feature_cols]
   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Find number of components that explain 95% of variance
    n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    y = df[category]
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, random_state=42, test_size=0.2, stratify=y
    )

    model = LogisticRegression(
        random_state=42, 
        max_iter=1000,  
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability for class 1

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print("\nPredicted Probabilities for Class 1:")
    print(y_prob[:10])  

    return accuracy, y_prob

if __name__ == "__main__":
    df = pd.read_csv("../../data/processed/music_features.csv")
    accuracy, probabilities = logistic_regression_classifier(df, "rock")
