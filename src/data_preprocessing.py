"""
Data Preprocessing Module for Twitter Bot Analysis
This module handles text cleaning, feature extraction, and preprocessing for the Twitter bot detection model.
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    print("NLTK resources may not be available. Some features might not work properly.")

class TwitterDataPreprocessor:
    """Class to preprocess Twitter data for bot detection."""
    
    def __init__(self, use_spacy=False):
        """
        Initialize the preprocessor.
        
        Args:
            use_spacy (bool): Whether to use spaCy for advanced NLP processing.
        """
        self.use_spacy = use_spacy
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Load spaCy model if requested
        if use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                print("spaCy model not found. Please install it using: python -m spacy download en_core_web_sm")
                self.use_spacy = False
    
    def clean_text(self, text):
        """
        Clean tweet text by removing URLs, mentions, hashtags, and special characters.
        
        Args:
            text (str): Raw tweet text.
            
        Returns:
            str: Cleaned text.
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """
        Tokenize and lemmatize text.
        
        Args:
            text (str): Cleaned text.
            
        Returns:
            list: List of lemmatized tokens.
        """
        if self.use_spacy:
            doc = self.nlp(text)
            tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        else:
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and token.isalpha()]
        
        return tokens
    
    def extract_user_features(self, df):
        """
        Extract user-level features that may indicate bot behavior.
        
        Args:
            df (pd.DataFrame): DataFrame containing tweet data.
            
        Returns:
            pd.DataFrame: DataFrame with user features.
        """
        # Group by author_id
        user_features = df.groupby('author_id').agg({
            'author_name': 'first',
            'author_username': 'first',
            'author_verified': 'first',
            'author_created_at': 'first',
            'author_followers': 'first',
            'author_following': 'first',
            'author_tweet_count': 'first',
            'tweet_id': 'count',  # Number of tweets in dataset
        }).reset_index()
        
        # Rename columns
        user_features.rename(columns={'tweet_id': 'dataset_tweet_count'}, inplace=True)
        
        # Calculate account age in days
        current_time = datetime.now()
        user_features['account_age_days'] = user_features['author_created_at'].apply(
            lambda x: (current_time - x).days if pd.notnull(x) else np.nan
        )
        
        # Calculate tweets per day
        user_features['tweets_per_day'] = user_features.apply(
            lambda row: row['author_tweet_count'] / row['account_age_days'] 
            if pd.notnull(row['account_age_days']) and row['account_age_days'] > 0 
            else np.nan, 
            axis=1
        )
        
        # Calculate follower-following ratio
        user_features['follower_following_ratio'] = user_features.apply(
            lambda row: row['author_followers'] / row['author_following'] 
            if pd.notnull(row['author_following']) and row['author_following'] > 0 
            else np.nan, 
            axis=1
        )
        
        return user_features
    
    def extract_tweet_features(self, df):
        """
        Extract tweet-level features for bot detection.
        
        Args:
            df (pd.DataFrame): DataFrame containing tweet data.
            
        Returns:
            pd.DataFrame: DataFrame with tweet features.
        """
        # Create a copy to avoid modifying the original
        tweet_features = df.copy()
        
        # Clean text
        tweet_features['cleaned_text'] = tweet_features['text'].apply(self.clean_text)
        
        # Text length features
        tweet_features['text_length'] = tweet_features['text'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        tweet_features['word_count'] = tweet_features['cleaned_text'].apply(
            lambda x: len(x.split()) if isinstance(x, str) else 0
        )
        
        # Engagement metrics
        tweet_features['engagement_score'] = tweet_features['retweet_count'] + \
                                            tweet_features['reply_count'] + \
                                            tweet_features['like_count'] + \
                                            tweet_features['quote_count']
        
        # Time features
        tweet_features['hour_of_day'] = tweet_features['created_at'].apply(
            lambda x: x.hour if pd.notnull(x) else np.nan
        )
        
        # Source device type (simplified)
        def categorize_source(source):
            if not isinstance(source, str):
                return 'unknown'
            source = source.lower()
            if 'android' in source:
                return 'android'
            elif 'iphone' in source or 'ios' in source:
                return 'ios'
            elif 'web' in source:
                return 'web'
            elif 'bot' in source or 'auto' in source:
                return 'automated'
            else:
                return 'other'
                
        tweet_features['source_type'] = tweet_features['source'].apply(categorize_source)
        
        return tweet_features
    
    def preprocess_data(self, df, extract_tokens=True):
        """
        Preprocess the entire dataset for analysis.
        
        Args:
            df (pd.DataFrame): Raw tweet DataFrame.
            extract_tokens (bool): Whether to extract and tokenize text.
            
        Returns:
            tuple: (processed_df, user_features)
        """
        # Extract tweet features
        processed_df = self.extract_tweet_features(df)
        
        # Extract user features
        user_features = self.extract_user_features(df)
        
        # Tokenize text if requested
        if extract_tokens:
            processed_df['tokens'] = processed_df['cleaned_text'].apply(self.tokenize_and_lemmatize)
        
        return processed_df, user_features
    
    def save_processed_data(self, processed_df, user_features, 
                           tweets_output='../data/processed_tweets.csv',
                           users_output='../data/user_features.csv'):
        """
        Save processed data to CSV files.
        
        Args:
            processed_df (pd.DataFrame): Processed tweet DataFrame.
            user_features (pd.DataFrame): User features DataFrame.
            tweets_output (str): Path to save processed tweets.
            users_output (str): Path to save user features.
        """
        # Save processed tweets
        processed_df.to_csv(tweets_output, index=False)
        print(f"Saved processed tweets to {tweets_output}")
        
        # Save user features
        user_features.to_csv(users_output, index=False)
        print(f"Saved user features to {users_output}")


if __name__ == "__main__":
    # Example usage
    try:
        # Load raw data
        raw_tweets = pd.read_csv('../data/caa_tweets.csv')
        
        # Initialize preprocessor
        preprocessor = TwitterDataPreprocessor(use_spacy=False)
        
        # Preprocess data
        processed_tweets, user_features = preprocessor.preprocess_data(raw_tweets)
        
        # Save processed data
        preprocessor.save_processed_data(processed_tweets, user_features)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have collected tweet data first using data_collection.py")