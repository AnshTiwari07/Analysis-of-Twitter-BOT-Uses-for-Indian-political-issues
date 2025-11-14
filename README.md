# Analysis of Twitter Bot Uses for Indian Political Issues (CAA)

An analysis project investigating the involvement of Twitter bots in the Citizenship Amendment Act (CAA) discourse in India. This project follows formal research guidelines and implements a Relevance Vector Machine (RVM) classifier to detect likely bot accounts, using data collected via Twitter APIs.

## Project Goals
- Collect recent tweets related to CAA/NRC using the Twitter API.
- Preprocess text and derive tweet/user features indicative of bot behavior.
- Train an RVM-based classifier and evaluate bot detection performance.
- Visualize bot vs human participation and content patterns in the CAA discussion.
- Provide a reproducible pipeline aligned with research best practices.

## Project Structure
```
Ml project/
├── data/                       # Raw and processed CSVs
├── models/                     # Saved trained models
├── notebooks/                  # Jupyter notebooks for exploration (optional)
├── results/                    # Generated plots and analysis artifacts
├── src/
│   ├── data_collection.py      # Collect tweets via Twitter API (CAA queries)
│   ├── data_preprocessing.py   # Clean text and derive tweet/user features
│   ├── rvm_classifier.py       # Train/evaluate RVM model, save outputs
│   └── analysis_visualization.py # Plots and comprehensive report
└── requirements.txt
```

## Prerequisites
- Python 3.9+ recommended
- A Twitter Developer account with API keys (v2 API)

Install dependencies:
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Note: First run may download NLTK corpora automatically. If it fails, you can manually run:
```
python -c "import nltk; [nltk.download(x) for x in ['punkt','stopwords','wordnet']]"
```

## Twitter API Credentials
You can provide credentials in two ways:

1) Environment variables
```
set TWITTER_CONSUMER_KEY=...
set TWITTER_CONSUMER_SECRET=...
set TWITTER_ACCESS_TOKEN=...
set TWITTER_ACCESS_TOKEN_SECRET=...
set TWITTER_BEARER_TOKEN=...
```

2) JSON file (recommended for local runs) at `credentials.json`:
```json
{
  "consumer_key": "...",
  "consumer_secret": "...",
  "access_token": "...",
  "access_token_secret": "...",
  "bearer_token": "..."
}
```
Then run scripts with: `python src/data_collection.py` and it will auto-detect.

## Usage

1) Collect CAA-related tweets (recent window)
```
python src/data_collection.py
```
- Saves to `data/caa_tweets.csv`.
- Uses a query targeting CAA/NRC terms and excludes retweets.
- Twitter API rate limits apply; ensure your keys are valid and comply with platform policies.

2) Preprocess and feature extraction
```
python src/data_preprocessing.py
```
- Produces `data/processed_tweets.csv` and `data/user_features.csv`.
- Derives text, engagement, temporal, and account features.

3) Train and evaluate the RVM classifier
```
python src/rvm_classifier.py
```
- Trains an RVM classifier on prepared features.
- Prints metrics and saves `models/rvm_bot_classifier.pkl`.
- Generates a feature relevance plot in `results/feature_relevance.png`.

4) Analysis and visualization report
```
python src/analysis_visualization.py
```
- Generates multiple plots in `results/`:
  - `bot_distribution.png`
  - `bot_activity_timeline.png`
  - `engagement_comparison.png`
  - `wordclouds.png`
  - `account_creation_timeline.png`
  - `topic_distribution.png`
- If no predictions file is present, it will create dummy predictions for demonstration.

## Research Considerations
- Clearly document the data collection window, query terms, and inclusion/exclusion criteria.
- Maintain reproducibility: pin dependency versions and save seeds/hyperparameters.
- Ethical use: respect Twitter’s terms, user privacy, and avoid identifying individuals. Focus on aggregate patterns.
- Limitations: Twitter API returns a recent slice; ensure your study states time bounds. RVM training can be slower and requires careful feature scaling.
- Labeling: The example uses synthetic labels (heuristics) only for demonstration. For publication-grade results, use a curated, validated labeled dataset of bots vs. humans.

## Troubleshooting
- Authentication errors: verify keys or `credentials.json` and confirm v2 API access.
- Rate limits/empty results: widen date range or adjust query; check API tier.
- spaCy model missing: run `python -m spacy download en_core_web_sm`.
- NLTK resource errors: run the NLTK download command above.
- RVM class availability: ensure `sklearn-rvm` is installed and supports classification (EMRVC/RVC). If `predict_proba` is unavailable, switch to `EMRVC` or use decision scores.

## Next Steps
- Replace dummy labels with a real annotated dataset.
- Expand features (network interactions, retweet graphs, temporal burstiness).
- Add notebook(s) for EDA and robustness checks.
- Consider comparing RVM with other classifiers (SVM, Logistic Regression, XGBoost) for benchmarking.

---

This project scaffolding is ready. Set up your credentials, install dependencies, and run the pipeline steps in order for initial results.