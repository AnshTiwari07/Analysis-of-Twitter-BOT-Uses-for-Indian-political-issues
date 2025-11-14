"""
Relevance Vector Machine (RVM) Classifier for Twitter Bot Detection
This module implements the RVM classifier for detecting Twitter bots in political discussions.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn_rvm import EMRVR
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

class TwitterBotClassifier:
    """Class for training and evaluating RVM classifier for Twitter bot detection."""
    
    def __init__(self):
        """Initialize the Twitter bot classifier."""
        self.model = None
        self.feature_names = None
        self.text_vectorizer = None
        self.scaler = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        
    def prepare_features(self, tweets_df, user_df, labeled_data=None):
        """
        Prepare features for the RVM classifier.
        
        Args:
            tweets_df (pd.DataFrame): Processed tweets DataFrame.
            user_df (pd.DataFrame): User features DataFrame.
            labeled_data (pd.DataFrame): Optional labeled data with 'is_bot' column.
            
        Returns:
            tuple: (X, y, feature_names) - features, labels, and feature names
        """
        # Merge tweet and user features
        merged_df = tweets_df.merge(user_df, on='author_id', how='left')
        
        # If labeled data is provided, merge with it
        if labeled_data is not None:
            merged_df = merged_df.merge(labeled_data[['author_id', 'is_bot']], 
                                       on='author_id', how='inner')
        else:
            # For demonstration, we'll use a heuristic to create synthetic labels
            # In a real scenario, you would use actual labeled data
            print("Warning: Using synthetic labels for demonstration purposes.")
            merged_df['is_bot'] = merged_df.apply(
                lambda row: 1 if ((row.get('tweets_per_day', 0) > 50) or 
                                 (row.get('follower_following_ratio', 0) < 0.01) or
                                 ('bot' in str(row.get('source', '')).lower())) else 0,
                axis=1
            )
        
        # Select features for classification
        text_features = ['cleaned_text']
        
        numeric_features = [
            'text_length', 'word_count', 'engagement_score',
            'tweets_per_day', 'account_age_days', 'follower_following_ratio',
            'dataset_tweet_count', 'author_followers', 'author_following'
        ]
        
        categorical_features = ['source_type', 'hour_of_day']
        
        # Filter out features that might not be available
        numeric_features = [f for f in numeric_features if f in merged_df.columns]
        categorical_features = [f for f in categorical_features if f in merged_df.columns]
        
        # Create feature matrix
        X = merged_df[numeric_features + categorical_features].copy()
        
        # Handle categorical features (one-hot encoding)
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        
        # Add TF-IDF features from text
        if 'cleaned_text' in merged_df.columns:
            self.text_vectorizer = TfidfVectorizer(max_features=100, min_df=5)
            text_features = self.text_vectorizer.fit_transform(
                merged_df['cleaned_text'].fillna('')
            )
            
            # Convert to DataFrame and add to features
            text_feature_names = [f'tfidf_{i}' for i in range(text_features.shape[1])]
            text_df = pd.DataFrame(text_features.toarray(), columns=text_feature_names)
            X = pd.concat([X.reset_index(drop=True), text_df], axis=1)
        
        # Get labels
        y = merged_df['is_bot'].values
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y, self.feature_names
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the RVM classifier.
        
        Args:
            X (pd.DataFrame): Feature matrix.
            y (np.array): Target labels.
            test_size (float): Proportion of data to use for testing.
            random_state (int): Random seed for reproducibility.
            
        Returns:
            tuple: (model, X_train, X_test, y_train, y_test) - trained model and data splits
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store splits for later analysis
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        print("Training RVM classifier...")
        try:
            from sklearn_rvm import EMRVC
            self.model = EMRVC(kernel='rbf')
        except Exception:
            self.model = EMRVR(kernel='rbf')
        self.model.fit(X_train_scaled, y_train)
        
        return self.model, X_train_scaled, X_test_scaled, y_train, y_test
    
    def plot_feature_relevance(self, output_path='../results/feature_relevance.png'):
        """
        Estimate feature importance via permutation importance on the held-out set.
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model has not been trained yet.")
        if self.X_test_scaled is None or self.y_test is None:
            raise ValueError("Missing test data for computing feature importance.")
        
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            self.model, self.X_test_scaled, self.y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        importances_mean = result.importances_mean
        
        # Align feature names length with importance length
        feature_names = self.feature_names
        if len(feature_names) != len(importances_mean):
            # If mismatch occurs due to unexpected preprocessing, truncate or pad accordingly
            min_len = min(len(feature_names), len(importances_mean))
            feature_names = feature_names[:min_len]
            importances_mean = importances_mean[:min_len]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances_mean
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Top 20 Features by Permutation Importance')
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Feature relevance plot saved to {output_path}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model.
        
        Args:
            X_test (np.array): Test features.
            y_test (np.array): Test labels.
            
        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        # Try to get probabilities; fallback if not available
        try:
            y_prob = self.model.predict_proba(X_test)[:, 1]
        except Exception:
            # Fallback: use a normalized decision function or y_pred
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(X_test)
                # Min-max normalize to [0,1]
                min_s, max_s = scores.min(), scores.max()
                y_prob = (scores - min_s) / (max_s - min_s + 1e-8)
            else:
                y_prob = y_pred.astype(float)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = None
        
        results = {
            'accuracy': report['accuracy'],
            'precision': report.get('1', {}).get('precision', None),
            'recall': report.get('1', {}).get('recall', None),
            'f1_score': report.get('1', {}).get('f1-score', None),
            'auc': auc,
            'confusion_matrix': conf_matrix,
            'classification_report': report
        }
        
        print(f"Accuracy: {results['accuracy']:.4f}")
        if results['precision'] is not None:
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"F1 Score: {results['f1_score']:.4f}")
        if auc is not None:
            print(f"AUC: {results['auc']:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return results
    
    def save_model(self, model_path='../models/rvm_bot_classifier.pkl'):
        """
        Save the trained model and preprocessing components.
        
        Args:
            model_path (str): Path to save the model.
        """
        if self.model is None:
            raise ValueError("No trained model to save.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and components
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'text_vectorizer': self.text_vectorizer
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='../models/rvm_bot_classifier.pkl'):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the saved model.
            
        Returns:
            self: The classifier instance with loaded model.
        """
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.scaler = model_data['scaler']
        self.text_vectorizer = model_data['text_vectorizer']
        
        print(f"Model loaded from {model_path}")
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (pd.DataFrame): Feature matrix.
            
        Returns:
            tuple: (y_pred, y_prob) - Predicted labels and probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
            
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        try:
            y_prob = self.model.predict_proba(X_scaled)[:, 1]
        except Exception:
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(X_scaled)
                min_s, max_s = scores.min(), scores.max()
                y_prob = (scores - min_s) / (max_s - min_s + 1e-8)
            else:
                y_prob = y_pred.astype(float)
        return y_pred, y_prob


if __name__ == "__main__":
    try:
        # Load processed data
        tweets_df = pd.read_csv('../data/processed_tweets.csv')
        user_df = pd.read_csv('../data/user_features.csv')
        
        # Initialize classifier
        classifier = TwitterBotClassifier()
        
        # Prepare features
        X, y, feature_names = classifier.prepare_features(tweets_df, user_df)
        
        # Train model
        model, X_train, X_test, y_train, y_test = classifier.train(X, y)
        
        # Evaluate model
        results = classifier.evaluate(X_test, y_test)
        
        # Plot feature relevance
        classifier.plot_feature_relevance()
        
        # Save model
        classifier.save_model()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have preprocessed the data first using data_preprocessing.py")