import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.LogisticRegression.logistic_regression import logistic_regression_classifier
from models.KNN.knn import knn_classify
from models.NaiveBayes.naive_bayes import naive_bayes_classify
from models.RandomForest.random_forest import random_forest_classify
from models.SVM.svm import svm_classify

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, hamming_loss

base_models = {
    "logistic_regression": logistic_regression_classifier,
    "knn": knn_classify,
    "naive_bayes": naive_bayes_classify,
    "random_forest": random_forest_classify,
    "svm": svm_classify
}


def call_classifier(classifier_name, *args, **kwargs):
    """
    Calls a specified classifier function with given arguments.
    
    Args:
        classifier_name (str): Name of the classifier to use (must be in function_map)
        *args: Positional arguments to pass to the classifier
        **kwargs: Keyword arguments to pass to the classifier
    
    Returns:
        The result of the classifier function
    """
    classifier_func = base_models[classifier_name]
    return classifier_func(*args, **kwargs)

N_CATEGORIES = 10

CATEGORIES = ["pop", "blues", "hip-hop","rock", "classical", "reggae", "country", "metal", "techno", "jazz"]

CLASSIFIERS = list(base_models.keys()) * 2

def stacking_model(df_music: pd.DataFrame):
    """
    Implements stacking ensemble for multi-label music genre classification.
    
    This model works in two phases:
    1. Base models make predictions for each genre
    2. A meta-model uses these predictions as features for final classification
    """
    df_music = df_music.drop(columns=["title"])
    # First split the data - we need this to avoid data leakage
    feature_cols = df_music.select_dtypes(include=[np.number]).columns.difference(CATEGORIES)
    X_original = df_music[feature_cols]
    y = df_music[CATEGORIES]
    
    # Split data into training and testing sets
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X_original, y, test_size=0.2, random_state=42, stratify=y[CATEGORIES[0]])  # Stratify by first genre
    
    # Initialize arrays to store base model predictions
    train_predictions = []
    test_predictions = []
    
    # Get predictions from each base classifier for each genre
    for genre, algo in zip(CATEGORIES, CLASSIFIERS):
        print(f"Training {algo} for {genre}")
        
        # Train the base classifier and get predictions
        try:
            # Get probabilities for training data
            _, train_prob = call_classifier(algo, X_train_orig, y_train[genre])
            # Get probabilities for test data using the same model
            _, test_prob = call_classifier(algo, X_test_orig, y_train[genre])
            
            train_predictions.append(train_prob)
            test_predictions.append(test_prob)
        except Exception as e:
            print(f"Error with classifier {algo} for {genre}: {str(e)}")
            continue
    
    # Stack predictions into feature matrices
    X_train_meta = np.column_stack(train_predictions)
    X_test_meta = np.column_stack(test_predictions)
    
    # Scale the meta-features
    scaler = StandardScaler()
    X_train_meta_scaled = scaler.fit_transform(X_train_meta)
    X_test_meta_scaled = scaler.transform(X_test_meta)
    
    # Train the meta-classifier
    print("\nTraining meta-classifier...")
    base_classifier = LogisticRegression(max_iter=1000)
    meta_model = MultiOutputClassifier(base_classifier)
    meta_model.fit(X_train_meta_scaled, y_train)
    
    # Make predictions with meta-classifier
    y_pred = meta_model.predict(X_test_meta_scaled)
    y_pred_proba = meta_model.predict_proba(X_test_meta_scaled)
    
    # Evaluate results
    print("\nClassification Report by Genre:")
    for i, genre in enumerate(CATEGORIES):
        print(f"\nMetrics for {genre}:")
        print(classification_report(y_test[genre], y_pred[:, i]))
    
    print(f"\nOverall Hamming Loss: {hamming_loss(y_test, y_pred):.4f}")
    
    # Show probabilities for first sample
    print("\nProbability distributions for first sample:")
    for i, genre in enumerate(CATEGORIES):
        prob = y_pred_proba[i][0][1]
        print(f"{genre}: {prob:.4f}")
    
    return meta_model, scaler

# Add a prediction function for new data
def predict_with_stacking(meta_model, scaler, base_models, new_data):
    """
    Makes predictions on new data using the stacked model.
    
    Args:
        meta_model: The trained meta-classifier
        scaler: The fitted scaler for meta-features
        base_models: Dictionary of trained base models
        new_data: Features for new samples
    """
    base_predictions = []
    
    # Get predictions from base models
    for genre, algo in zip(CATEGORIES, CLASSIFIERS):
        _, probs = base_models[algo].predict_proba(new_data)
        base_predictions.append(probs)
    
    # Stack predictions
    X_meta = np.column_stack(base_predictions)
    X_meta_scaled = scaler.transform(X_meta)
    
    # Make final predictions
    return meta_model.predict(X_meta_scaled)

if __name__ == "__main__":

    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data/processed/music_features_binary_genres.csv"

   
    df_music = pd.read_csv(csv_path)
    meta_model, scaler = stacking_model(df_music)

    # Make predictions on new data
    # new_predictions = predict_with_stacking(meta_model, scaler, base_models, new_data)