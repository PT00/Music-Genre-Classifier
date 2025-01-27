# helper libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import dump, load



# inner files
from models.KNN.knn import knn_classify
from models.NaiveBayes.naive_bayes import naive_bayes_classify
from models.LogisticRegression.logistic_regression import logistic_regression_classifier
from models.RandomForest.random_forest import random_forest_classify
from models.SVM.svm import svm_classify
from scripts.config import DATAFRAME_PATH, PATH_TO_BINARY_MODELS, PATH_TO_LAST_STEP_MODELS
from .logictic_regression_predict import train_logistic_regression
from scripts.extract_features import extract_audio_features
from .create_log import log_classification_results
from scripts.config import DATAFRAME_PATH




# RUN AS MODULE python -m predict.logictic_regression_predict

class Pipeline:
    
    #, 'pop', 'classical', 'blues', 'hip-hop', 'reggae', 'country', 'metal', 'techno', 'jazz'
    def __init__(self, df_music):
        self.CATEGORIES = ['rock', 'pop', 'classical', 'blues', 'hip-hop', 'reggae', 'country', 'metal', 'techno', 'jazz']
        self.MODELS_NAMES = ["knn", "lr", "nb", "rf", "svm", "xgboost"] # arrays used for models loading
        self.MODELS_FUN = [knn_classify, logistic_regression_classifier, naive_bayes_classify, random_forest_classify, svm_classify, ]
        
        self.df_music = df_music
        # dictionary of models
        self.models = {}
        # dictionary for regression
        self.logistic_regressions = {}

        self.train_pipeline()
    
    def train_pipeline(self):
        for category in tqdm(self.CATEGORIES):
            # for each category we train a model, 
            # create matrix of probabilities and create a vector of true labels
            X, y = self.train_all_models_for_category(category)

            self.models[category] = {
                'feature_matrix': X,
                'labels': y
            }
            
            # for each category, based on vectors from models
            # we train logistic regression
            # category is passed solely for organising
            # print(self.models["rock"]["feature_matrix"].shape)
            # print("last step logistic regression trained on: ", X.shape)
            # add verbose=1 for logging
            train_logistic_regression(X, y, category, model_names=self.MODELS_NAMES, verbose=1)
     


    """returns trained models, matrix with probabilities form each model and true labels"""
    def train_all_models_for_category(self, category: str):
        # matrix [test_size]x[#models]
        # for each of 200 songs (songs from test set for each model)
        # we gather probabilities for it being from 'category'
        # resulting matrix is 200x[#models] -> 200 samples with #models features
        songs_probs_matrix = []
        trained_models = []

        # here we run models for each category and save them to reuse
        for model_fun in self.MODELS_FUN:
            
            probabilities, y = model_fun(self.df_music, category)
            
            # build vector of probabilities, value being =1
            songs_probs_matrix.append(probabilities[:, 1])
        # each model uses the same sample to calcualte test, so y labels are the same
        return np.column_stack(songs_probs_matrix), y.values


    def classify_song(self, song_path: str):
        
        # make sure values are in the same order as in trained CSV
        # creates a vector (1,37) that can be fed to the model
        features = np.array([float(num) for num in extract_audio_features(song_path).values()]).reshape(1, -1)
        results = {}
        

        for category in self.CATEGORIES:
            
            # load last step logistic regression
            logistic = load(f'{PATH_TO_LAST_STEP_MODELS}/categorized_regression/{category}.joblib')
            
            # vector build according to binary models predictions
            song_probs_vector = []
            for name in self.MODELS_NAMES:

                # Load saved components
                model = load(f'{PATH_TO_BINARY_MODELS}/{name}/model_{category}.joblib')
                scaler = load(f'{PATH_TO_BINARY_MODELS}/{name}/scaler_{category}.joblib')
            
                # Transform features
                X_scaled = scaler.transform(features)
                # X_pca = pca.transform(X_scaled)
                
                # biorę PPB na 1 z każdego modelu, w kolejnych kategoriach
                probs = model.predict_proba(X_scaled)[:, 1][0]
                print("probs", probs)
                song_probs_vector.append(probs)
            

            # last step logistic regression, accepts a vector created by binary models
            # print("Input for last step logistic regression: ", np.array([song_probs_vector]).shape)
            logistic_prob = logistic.predict_proba(np.array([song_probs_vector]))[:, 1][0]
            results[category] = logistic_prob

            
            
        return results


if __name__ == "__main__":
    df_music = pd.read_csv(DATAFRAME_PATH)
    pipeline = Pipeline(df_music)