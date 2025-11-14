"""
Main orchestrator for CAA Twitter bot analysis.
Runs the pipeline end-to-end: data collection (or synthetic fallback), preprocessing,
RVM training + predictions, and analysis visualizations.
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Allow running from src
dir_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(dir_path)

# Local imports
from data_collection import TwitterDataCollector
from data_preprocessing import TwitterDataPreprocessor
from rvm_classifier import TwitterBotClassifier
from analysis_visualization import TwitterBotAnalyzer


def ensure_dirs():
    for d in [os.path.join(root_path, 'data'), os.path.join(root_path, 'models'), os.path.join(root_path, 'results')]:
        os.makedirs(d, exist_ok=True)


def synthesize_caa_tweets(n_users=150, n_tweets=800):
    np.random.seed(42)
    start = datetime.now() - timedelta(days=20)
    texts = [
        "CAA protests in India spark debate over citizenship.",
        "Support for CAA grows amid confusion about NRC.",
        "Citizenship Amendment Act and NRC policy discussion.",
        "Voices against CAA rise; what are the implications?",
        "NRC and CAA explained; is it fair for all?",
        "Debate on CAA continues across cities; protests and support.",
        "CAA policy merits and drawbacks discussed in parliament.",
        "CAA/NRC policy clarification by ministers sparks reactions.",
        "India politics and CAA: perspectives from different states.",
        "Why is NRC linked to CAA in discourse?",
    ]
    sources = ["Twitter Web App", "Twitter for iPhone", "Twitter for Android", "BotService"]

    user_ids = np.arange(100000, 100000 + n_users)
    user_rows = []
    for uid in user_ids:
        created = start - timedelta(days=np.random.randint(50, 4000))
        followers = np.random.randint(10, 50000)
        following = np.random.randint(1, 5000)
        tweets = np.random.randint(50, 200000)
        verified = np.random.rand() < 0.05
        user_rows.append({
            'author_id': uid,
            'author_name': f'User{uid}',
            'author_username': f'user_{uid}',
            'author_verified': verified,
            'author_created_at': created,
            'author_followers': followers,
            'author_following': following,
            'author_tweet_count': tweets,
        })

    tweets_rows = []
    for _ in range(n_tweets):
        uid = int(np.random.choice(user_ids))
        text = np.random.choice(texts)
        dt = start + timedelta(days=np.random.randint(0, 20), hours=np.random.randint(0, 24))
        metrics = {
            'retweet_count': np.random.poisson(2),
            'reply_count': np.random.poisson(1),
            'like_count': np.random.poisson(5),
            'quote_count': np.random.poisson(0.5),
        }
        src = np.random.choice(sources, p=[0.3, 0.25, 0.35, 0.1])
        lang = 'en'
        tweets_rows.append({
            'tweet_id': int(np.random.randint(10_000_000, 99_999_999)),
            'text': text,
            'created_at': dt,
            **metrics,
            'source': src,
            'lang': lang,
            'author_id': uid,
        })

    df_tweets = pd.DataFrame(tweets_rows)
    df_users = pd.DataFrame(user_rows)
    df = df_tweets.merge(df_users, on='author_id', how='left')
    return df


def collect_or_synthesize():
    out_path = os.path.join(root_path, 'data', 'caa_tweets.csv')
    try:
        print("Attempting Twitter API collection...")
        creds_json = os.path.join(root_path, 'credentials.json')
        collector = TwitterDataCollector(credentials_file=creds_json if os.path.exists(creds_json) else None)
        tweets_df = collector.collect_caa_tweets(max_results=1000, days_back=14)
        collector.save_tweets(tweets_df, output_file=out_path)
        print(f"Collected {len(tweets_df)} tweets via API.")
    except Exception as e:
        print(f"API collection failed: {e}\nFalling back to synthetic dataset.")
        df = synthesize_caa_tweets()
        # Ensure datetime serialization for CSV
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['author_created_at'] = pd.to_datetime(df['author_created_at'])
        df.to_csv(out_path, index=False)
        print(f"Synthetic dataset saved to {out_path} with {len(df)} rows.")


def preprocess():
    print("Preprocessing data...")
    raw_path = os.path.join(root_path, 'data', 'caa_tweets.csv')
    tweets_df = pd.read_csv(raw_path, parse_dates=['created_at', 'author_created_at'], infer_datetime_format=True)
    pre = TwitterDataPreprocessor(use_spacy=False)
    processed_tweets, user_features = pre.preprocess_data(tweets_df)
    pre.save_processed_data(processed_tweets, user_features,
                            tweets_output=os.path.join(root_path, 'data', 'processed_tweets.csv'),
                            users_output=os.path.join(root_path, 'data', 'user_features.csv'))


def train_and_predict():
    print("Training classifier...")
    tweets_df = pd.read_csv(os.path.join(root_path, 'data', 'processed_tweets.csv'))
    user_df = pd.read_csv(os.path.join(root_path, 'data', 'user_features.csv'))

    clf = TwitterBotClassifier()
    X, y, feature_names = clf.prepare_features(tweets_df, user_df)
    model, X_train, X_test, y_train, y_test = clf.train(X, y)
    results = clf.evaluate(X_test, y_test)
    clf.plot_feature_relevance(output_path=os.path.join(root_path, 'results', 'feature_relevance.png'))
    clf.save_model(model_path=os.path.join(root_path, 'models', 'rvm_bot_classifier.pkl'))

    # Create predictions per row and aggregate per author
    print("Generating predictions...")
    y_pred, y_prob = clf.predict(X)

    # Rebuild merge to get author_id alignment
    merged_df = tweets_df.merge(user_df, on='author_id', how='left')
    preds_df = pd.DataFrame({
        'author_id': merged_df['author_id'].values,
        'is_bot_row': y_pred,
        'bot_probability_row': y_prob,
    })
    agg = preds_df.groupby('author_id').agg({
        'bot_probability_row': 'mean',
        'is_bot_row': lambda s: int(np.round(s.mean() >= 0.5)),
    }).reset_index().rename(columns={
        'bot_probability_row': 'bot_probability',
        'is_bot_row': 'is_bot',
    })

    out_pred = os.path.join(root_path, 'data', 'bot_predictions.csv')
    agg.to_csv(out_pred, index=False)
    print(f"Saved aggregated predictions to {out_pred}")


def visualize():
    print("Generating analysis and visualizations...")
    analyzer = TwitterBotAnalyzer()
    analyzer.load_data(
        tweets_path=os.path.join(root_path, 'data', 'processed_tweets.csv'),
        user_path=os.path.join(root_path, 'data', 'user_features.csv'),
        predictions_path=os.path.join(root_path, 'data', 'bot_predictions.csv'),
    )
    analyzer.generate_comprehensive_report()


if __name__ == '__main__':
    try:
        ensure_dirs()
        collect_or_synthesize()
        preprocess()
        train_and_predict()
        visualize()
        print("Pipeline completed successfully.")
    except Exception:
        print("Pipeline failed. Traceback:")
        traceback.print_exc()
        sys.exit(1)