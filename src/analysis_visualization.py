"""
Analysis and Visualization Module for Twitter Bot Analysis
This module provides functions for analyzing and visualizing Twitter bot activity in political discussions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from datetime import datetime

class TwitterBotAnalyzer:
    """Class for analyzing and visualizing Twitter bot activity."""
    
    def __init__(self, tweets_df=None, user_df=None, predictions_df=None):
        self.tweets_df = tweets_df
        self.user_df = user_df
        self.predictions_df = predictions_df
        self.results_dir = '../results'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_data(self, tweets_path='../data/processed_tweets.csv', 
                 user_path='../data/user_features.csv',
                 predictions_path=None):
        self.tweets_df = pd.read_csv(tweets_path)
        self.user_df = pd.read_csv(user_path)
        if predictions_path and os.path.exists(predictions_path):
            self.predictions_df = pd.read_csv(predictions_path)
        else:
            self.predictions_df = None
            print("No predictions file provided or found.")
    
    def merge_data_with_predictions(self):
        if self.tweets_df is None or self.user_df is None:
            raise ValueError("Tweets or user data not loaded.")
        merged_df = self.tweets_df.merge(self.user_df, on='author_id', how='left')
        if self.predictions_df is not None:
            merged_df = merged_df.merge(
                self.predictions_df[['author_id', 'is_bot', 'bot_probability']], 
                on='author_id', how='left'
            )
        return merged_df
    
    def plot_bot_distribution(self, output_path=None):
        if self.predictions_df is None:
            print("Predictions not loaded; skipping bot distribution plot.")
            return
        bot_counts = self.predictions_df['is_bot'].value_counts()
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=bot_counts.index.astype(str), y=bot_counts.values)
        plt.title('Bots vs. Humans in CAA Discussion')
        plt.xlabel('Account Type (0=Human, 1=Bot)')
        plt.ylabel('Count')
        for i, count in enumerate(bot_counts.values):
            ax.text(i, count + 0.5, f"{count}", ha='center')
        out = output_path or f"{self.results_dir}/bot_distribution.png"
        plt.savefig(out)
        plt.close()
        print(f"Bot distribution plot saved to {out}")
    
    def plot_bot_activity_timeline(self, output_path=None):
        merged_df = self.merge_data_with_predictions()
        if 'created_at' in merged_df.columns and not pd.api.types.is_datetime64_any_dtype(merged_df['created_at']):
            merged_df['created_at'] = pd.to_datetime(merged_df['created_at'])
        merged_df['date'] = merged_df['created_at'].dt.date
        activity_by_date = merged_df.groupby(['date', 'is_bot']).size().reset_index(name='count')
        pivot = activity_by_date.pivot(index='date', columns='is_bot', values='count').fillna(0)
        if 1 not in pivot.columns:
            pivot[1] = 0
        if 0 not in pivot.columns:
            pivot[0] = 0
        pivot.columns = ['Human', 'Bot']
        plt.figure(figsize=(14, 8))
        pivot.plot(kind='line', ax=plt.gca(), marker='o')
        plt.title('Bot vs. Human Activity Timeline')
        plt.xlabel('Date')
        plt.ylabel('Tweets')
        plt.legend(['Human', 'Bot'])
        plt.grid(True, alpha=0.3)
        out = output_path or f"{self.results_dir}/bot_activity_timeline.png"
        plt.savefig(out)
        plt.close()
        print(f"Bot activity timeline plot saved to {out}")
    
    def plot_engagement_comparison(self, output_path=None):
        merged_df = self.merge_data_with_predictions()
        metrics = ['retweet_count', 'reply_count', 'like_count', 'quote_count']
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        for i, metric in enumerate(metrics):
            if metric in merged_df.columns:
                sns.boxplot(x='is_bot', y=metric, data=merged_df, ax=axes[i])
                axes[i].set_title(f'{metric.replace("_", " ").title()} by Account Type')
                axes[i].set_xlabel('Account Type (0=Human, 1=Bot)')
                axes[i].set_ylabel(metric.replace('_', ' ').title())
        plt.tight_layout()
        out = output_path or f"{self.results_dir}/engagement_comparison.png"
        plt.savefig(out)
        plt.close()
        print(f"Engagement comparison plot saved to {out}")
    
    def generate_wordclouds(self, output_path=None):
        merged_df = self.merge_data_with_predictions()
        if 'cleaned_text' not in merged_df.columns:
            print('No cleaned_text found; skipping wordclouds.')
            return
        bot_tweets = ' '.join(merged_df[merged_df['is_bot'] == 1]['cleaned_text'].fillna('').astype(str))
        human_tweets = ' '.join(merged_df[merged_df['is_bot'] == 0]['cleaned_text'].fillna('').astype(str))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        bot_wc = WordCloud(width=800, height=400, background_color='white').generate(bot_tweets)
        human_wc = WordCloud(width=800, height=400, background_color='white').generate(human_tweets)
        ax1.imshow(bot_wc, interpolation='bilinear'); ax1.set_title('Bot Tweet Content'); ax1.axis('off')
        ax2.imshow(human_wc, interpolation='bilinear'); ax2.set_title('Human Tweet Content'); ax2.axis('off')
        plt.tight_layout()
        out = output_path or f"{self.results_dir}/wordclouds.png"
        plt.savefig(out)
        plt.close()
        print(f"Wordclouds saved to {out}")
    
    def plot_account_creation_timeline(self, output_path=None):
        merged_df = self.merge_data_with_predictions()
        if 'author_created_at' not in merged_df.columns:
            print('No author_created_at found; skipping account creation timeline.')
            return
        if not pd.api.types.is_datetime64_any_dtype(merged_df['author_created_at']):
            merged_df['author_created_at'] = pd.to_datetime(merged_df['author_created_at'])
        merged_df['creation_date'] = merged_df['author_created_at'].dt.date
        creation_by_date = merged_df.groupby(['creation_date', 'is_bot']).size().reset_index(name='count')
        pivot = creation_by_date.pivot(index='creation_date', columns='is_bot', values='count').fillna(0).cumsum()
        if 1 not in pivot.columns:
            pivot[1] = 0
        if 0 not in pivot.columns:
            pivot[0] = 0
        pivot.columns = ['Human', 'Bot']
        plt.figure(figsize=(14, 8))
        pivot.plot(kind='line', ax=plt.gca())
        plt.title('Cumulative Account Creation: Bots vs. Humans')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Accounts')
        plt.legend(['Human', 'Bot'])
        plt.grid(True, alpha=0.3)
        out = output_path or f"{self.results_dir}/account_creation_timeline.png"
        plt.savefig(out)
        plt.close()
        print(f"Account creation timeline plot saved to {out}")
    
    def generate_comprehensive_report(self):
        print("Generating comprehensive report...")
        self.plot_bot_distribution()
        self.plot_bot_activity_timeline()
        self.plot_engagement_comparison()
        self.generate_wordclouds()
        self.plot_account_creation_timeline()
        print("Comprehensive report generated.")

if __name__ == "__main__":
    analyzer = TwitterBotAnalyzer()
    analyzer.load_data()
    analyzer.generate_comprehensive_report()