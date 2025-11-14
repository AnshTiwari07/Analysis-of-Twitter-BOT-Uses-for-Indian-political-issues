"""
Twitter Data Collection Module for CAA Political Issue Analysis
This module handles the collection of tweets related to CAA political issues in India.
"""

import os
import json
import tweepy
import pandas as pd
from datetime import datetime, timedelta

class TwitterDataCollector:
    """Class to collect Twitter data related to CAA political issues."""
    
    def __init__(self, credentials_file=None):
        """
        Initialize the Twitter data collector.
        
        Args:
            credentials_file (str): Path to Twitter API credentials JSON file.
                                   If None, will look for environment variables.
        """
        self.api = self._authenticate(credentials_file)
        
    def _authenticate(self, credentials_file):
        """
        Authenticate with Twitter API.
        
        Args:
            credentials_file (str): Path to credentials file.
            
        Returns:
            tweepy.API: Authenticated API object.
        """
        if credentials_file and os.path.exists(credentials_file):
            with open(credentials_file, 'r') as f:
                creds = json.load(f)
                consumer_key = creds.get('consumer_key')
                consumer_secret = creds.get('consumer_secret')
                access_token = creds.get('access_token')
                access_token_secret = creds.get('access_token_secret')
                bearer_token = creds.get('bearer_token')
        else:
            # Try to get from environment variables
            consumer_key = os.environ.get('TWITTER_CONSUMER_KEY')
            consumer_secret = os.environ.get('TWITTER_CONSUMER_SECRET')
            access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
            access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')
            bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
            
        if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
            if not bearer_token:
                raise ValueError("Twitter API credentials not found.")
            
        # Use v2 API with bearer token
        client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=consumer_key, 
            consumer_secret=consumer_secret,
            access_token=access_token, 
            access_token_secret=access_token_secret
        )
        
        return client
    
    def collect_tweets(self, query, max_results=100, start_time=None, end_time=None):
        """
        Collect tweets based on search query.
        
        Args:
            query (str): Search query string.
            max_results (int): Maximum number of tweets to collect.
            start_time (datetime): Start time for tweet search.
            end_time (datetime): End time for tweet search.
            
        Returns:
            pd.DataFrame: DataFrame containing collected tweets.
        """
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
            
        if not end_time:
            end_time = datetime.now()
            
        # Format dates for Twitter API
        start_time = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Define tweet fields to retrieve
        tweet_fields = ['created_at', 'author_id', 'public_metrics', 'source', 'lang']
        user_fields = ['name', 'username', 'description', 'public_metrics', 'verified', 'created_at']
        
        # Collect tweets
        tweets = []
        
        # Paginate through results
        pagination_token = None
        collected = 0
        
        while collected < max_results:
            try:
                response = self.api.search_recent_tweets(
                    query=query,
                    max_results=min(100, max_results - collected),  # API limit is 100 per request
                    tweet_fields=tweet_fields,
                    user_fields=user_fields,
                    expansions=['author_id'],
                    start_time=start_time,
                    end_time=end_time,
                    next_token=pagination_token
                )
                
                if not response.data:
                    break
                    
                # Process tweets and users
                users = {user.id: user for user in response.includes['users']}
                
                for tweet in response.data:
                    author = users.get(tweet.author_id)
                    tweet_data = {
                        'tweet_id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'reply_count': tweet.public_metrics['reply_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'quote_count': tweet.public_metrics['quote_count'],
                        'source': tweet.source,
                        'lang': tweet.lang,
                        'author_id': tweet.author_id,
                        'author_name': author.name if author else None,
                        'author_username': author.username if author else None,
                        'author_verified': author.verified if author else None,
                        'author_created_at': author.created_at if author else None,
                        'author_followers': author.public_metrics['followers_count'] if author else None,
                        'author_following': author.public_metrics['following_count'] if author else None,
                        'author_tweet_count': author.public_metrics['tweet_count'] if author else None
                    }
                    tweets.append(tweet_data)
                
                collected += len(response.data)
                
                # Get next pagination token
                pagination_token = response.meta.get('next_token')
                if not pagination_token:
                    break
                    
            except Exception as e:
                print(f"Error collecting tweets: {e}")
                break
                
        # Convert to DataFrame
        df = pd.DataFrame(tweets)
        return df
    
    def collect_caa_tweets(self, max_results=1000, days_back=30):
        """
        Collect tweets specifically related to CAA political issues.
        
        Args:
            max_results (int): Maximum number of tweets to collect.
            days_back (int): Number of days to look back.
            
        Returns:
            pd.DataFrame: DataFrame containing collected tweets.
        """
        # Define search query for CAA-related tweets
        caa_query = '(CAA OR "Citizenship Amendment Act" OR NRC OR "National Register of Citizens") (India OR Modi OR BJP OR protest OR politics) -is:retweet lang:en'
        
        # Set time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Collect tweets
        return self.collect_tweets(
            query=caa_query,
            max_results=max_results,
            start_time=start_time,
            end_time=end_time
        )
    
    def save_tweets(self, tweets_df, output_file='../data/caa_tweets.csv'):
        """
        Save collected tweets to a CSV file.
        
        Args:
            tweets_df (pd.DataFrame): DataFrame containing tweets.
            output_file (str): Path to output file.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to CSV
        tweets_df.to_csv(output_file, index=False)
        print(f"Saved {len(tweets_df)} tweets to {output_file}")


if __name__ == "__main__":
    # Example usage
    collector = TwitterDataCollector()
    try:
        tweets = collector.collect_caa_tweets(max_results=500)
        collector.save_tweets(tweets)
    except Exception as e:
        print(f"Error: {e}")
        print("Note: You need to set up Twitter API credentials to use this module.")
        print("Either create a credentials.json file or set environment variables.")