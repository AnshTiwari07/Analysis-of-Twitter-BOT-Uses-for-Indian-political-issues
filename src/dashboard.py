#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dashboard for Twitter Bot Detection and Monitoring
-------------------------------------------------
This script creates an interactive dashboard for real-time monitoring of Twitter bot activity
related to Indian political discussions, particularly around the Citizenship Amendment Act (CAA).
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import base64
from io import BytesIO

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from data_preprocessing import TwitterDataPreprocessor
from rvm_classifier import TwitterBotClassifier

# Set paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

# Ensure directories exist
for directory in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize the Dash app
app = dash.Dash(__name__, title="Twitter Bot Monitor", 
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

# Define colors
colors = {
    'background': '#F9F9F9',
    'text': '#333333',
    'bot': '#FF5733',
    'human': '#33A1FD',
    'grid': '#DDDDDD',
    'accent': '#9C27B0'
}

# Helper functions
def load_data():
    """Load and prepare data for the dashboard."""
    try:
        # Load processed data
        tweets_df = pd.read_csv(os.path.join(DATA_DIR, 'processed_tweets.csv'))
        user_df = pd.read_csv(os.path.join(DATA_DIR, 'user_features.csv'))
        
        # Load predictions if available
        try:
            predictions_df = pd.read_csv(os.path.join(DATA_DIR, 'bot_predictions.csv'))
        except FileNotFoundError:
            # Create synthetic predictions for demonstration
            print("Bot predictions not found. Creating synthetic data for demonstration.")
            user_ids = user_df['author_id'].unique()
            predictions_df = pd.DataFrame({
                'author_id': user_ids,
                'is_bot': np.random.choice([0, 1], size=len(user_ids), p=[0.7, 0.3]),
                'bot_probability': np.random.beta(2, 5, size=len(user_ids))
            })
        
        # Merge datasets
        merged_df = tweets_df.merge(user_df, on='author_id', how='left')
        merged_df = merged_df.merge(predictions_df[['author_id', 'is_bot', 'bot_probability']], 
                                   on='author_id', how='left')
        
        # Fill missing values
        merged_df['is_bot'] = merged_df['is_bot'].fillna(0)
        merged_df['bot_probability'] = merged_df['bot_probability'].fillna(0)
        
        # Convert timestamp if available
        if 'created_at' in merged_df.columns:
            merged_df['created_at'] = pd.to_datetime(merged_df['created_at'])
            merged_df['date'] = merged_df['created_at'].dt.date
            merged_df['hour'] = merged_df['created_at'].dt.hour
        
        return merged_df, tweets_df, user_df, predictions_df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return empty dataframes if data loading fails
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def generate_wordcloud_image(text):
    """Generate a wordcloud image from text."""
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         max_words=100, contour_width=3).generate(text)
    img = BytesIO()
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

# Load data
merged_df, tweets_df, user_df, predictions_df = load_data()

# App layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'padding': '20px'}, children=[
    # Header
    html.Div([
        html.H1("Twitter Bot Detection Dashboard", 
                style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '20px'}),
        html.P("Real-time monitoring of Twitter bot activity related to Indian political discussions",
               style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '30px'})
    ]),
    
    # Summary statistics
    html.Div([
        html.H2("Summary Statistics", style={'color': colors['text']}),
        html.Div([
            html.Div([
                html.H3("Total Accounts", style={'textAlign': 'center'}),
                html.P(f"{len(user_df)}", style={'textAlign': 'center', 'fontSize': '2em'})
            ], className='stat-box', style={'backgroundColor': 'white', 'borderRadius': '10px', 
                                           'padding': '20px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)',
                                           'width': '30%', 'display': 'inline-block', 'margin': '10px'}),
            html.Div([
                html.H3("Bot Accounts", style={'textAlign': 'center'}),
                html.P(f"{predictions_df['is_bot'].sum()}", style={'textAlign': 'center', 'fontSize': '2em', 'color': colors['bot']})
            ], className='stat-box', style={'backgroundColor': 'white', 'borderRadius': '10px', 
                                           'padding': '20px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)',
                                           'width': '30%', 'display': 'inline-block', 'margin': '10px'}),
            html.Div([
                html.H3("Total Tweets", style={'textAlign': 'center'}),
                html.P(f"{len(tweets_df)}", style={'textAlign': 'center', 'fontSize': '2em'})
            ], className='stat-box', style={'backgroundColor': 'white', 'borderRadius': '10px', 
                                           'padding': '20px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)',
                                           'width': '30%', 'display': 'inline-block', 'margin': '10px'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px'})
    ]),
    
    # Bot Distribution
    html.Div([
        html.H2("Bot Distribution", style={'color': colors['text']}),
        dcc.Graph(
            id='bot-distribution',
            figure=px.pie(
                predictions_df, 
                names=predictions_df['is_bot'].map({0: 'Human', 1: 'Bot'}),
                color=predictions_df['is_bot'].map({0: 'Human', 1: 'Bot'}),
                color_discrete_map={'Human': colors['human'], 'Bot': colors['bot']},
                title='Distribution of Bots vs Humans'
            ).update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font={'color': colors['text']}
            )
        )
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 
              'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 'marginBottom': '30px'}),
    
    # Bot Probability Distribution
    html.Div([
        html.H2("Bot Probability Distribution", style={'color': colors['text']}),
        dcc.Graph(
            id='bot-probability',
            figure=px.histogram(
                predictions_df, 
                x='bot_probability',
                nbins=30,
                title='Distribution of Bot Probability Scores',
                labels={'bot_probability': 'Bot Probability'},
                color_discrete_sequence=[colors['accent']]
            ).update_layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font={'color': colors['text']}
            ).add_vline(
                x=0.5, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Decision Threshold",
                annotation_position="top right"
            )
        )
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 
              'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 'marginBottom': '30px'}),
    
    # Temporal Analysis
    html.Div([
        html.H2("Temporal Analysis", style={'color': colors['text']}),
        html.P("Select date range to analyze bot activity over time:"),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=merged_df['created_at'].min() if 'created_at' in merged_df.columns else datetime.datetime.now() - datetime.timedelta(days=7),
            end_date=merged_df['created_at'].max() if 'created_at' in merged_df.columns else datetime.datetime.now(),
            display_format='YYYY-MM-DD'
        ),
        dcc.Graph(id='temporal-analysis')
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 
              'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 'marginBottom': '30px'}),
    
    # Word Clouds
    html.Div([
        html.H2("Content Analysis", style={'color': colors['text']}),
        html.Div([
            html.Div([
                html.H3("Bot Tweet Word Cloud", style={'textAlign': 'center'}),
                html.Img(id='bot-wordcloud', src=(
                    f"data:image/png;base64,{generate_wordcloud_image(' '.join(merged_df[merged_df['is_bot'] == 1]['cleaned_text'].fillna('')))}"
                    if 'cleaned_text' in merged_df.columns else ''
                ), style={'width': '100%'})
            ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                html.H3("Human Tweet Word Cloud", style={'textAlign': 'center'}),
                html.Img(id='human-wordcloud', src=(
                    f"data:image/png;base64,{generate_wordcloud_image(' '.join(merged_df[merged_df['is_bot'] == 0]['cleaned_text'].fillna('')))}"
                    if 'cleaned_text' in merged_df.columns else ''
                ), style={'width': '100%'})
            ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
        ])
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 
              'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 'marginBottom': '30px'}),
    
    # Real-time Monitoring Controls
    html.Div([
        html.H2("Real-time Monitoring", style={'color': colors['text']}),
        html.P("Configure real-time monitoring parameters:"),
        html.Div([
            html.Label("Refresh Interval (seconds):"),
            dcc.Slider(
                id='refresh-interval',
                min=30,
                max=300,
                step=30,
                value=60,
                marks={i: f'{i}s' for i in range(30, 301, 30)}
            )
        ], style={'marginBottom': '20px'}),
        html.Div([
            html.Button('Start Monitoring', id='start-monitoring', n_clicks=0, 
                       style={'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 
                              'padding': '10px 20px', 'borderRadius': '5px', 'marginRight': '10px'}),
            html.Button('Stop Monitoring', id='stop-monitoring', n_clicks=0,
                       style={'backgroundColor': '#f44336', 'color': 'white', 'border': 'none', 
                              'padding': '10px 20px', 'borderRadius': '5px'})
        ]),
        dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0, disabled=True),
        html.Div(id='monitoring-status', style={'marginTop': '10px'})
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 
              'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 'marginBottom': '30px'}),
    
    # Footer
    html.Div([
        html.P("Twitter Bot Detection for Indian Political Issues - Dashboard",
               style={'textAlign': 'center', 'color': colors['text'], 'padding': '20px'})
    ])
])

# Callbacks
@app.callback(
    Output('temporal-analysis', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_temporal_analysis(start_date, end_date):
    if 'created_at' not in merged_df.columns or start_date is None or end_date is None:
        return go.Figure().update_layout(
            title="No temporal data available",
            xaxis_title="Date",
            yaxis_title="Number of Tweets"
        )

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    filtered_df = merged_df[(merged_df['created_at'] >= start_dt) & (merged_df['created_at'] <= end_dt)]

    if filtered_df.empty:
        return go.Figure().update_layout(
            title="No data in selected range",
            xaxis_title="Date",
            yaxis_title="Number of Tweets"
        )

    date_counts = filtered_df.groupby([pd.Grouper(key='created_at', freq='D'), 'is_bot']).size().reset_index(name='count')
    date_counts['is_bot'] = date_counts['is_bot'].map({0: 'Human', 1: 'Bot'})

    fig = px.line(
        date_counts, 
        x='created_at', 
        y='count', 
        color='is_bot',
        title='Bot vs Human Activity Over Time',
        labels={'created_at': 'Date', 'count': 'Number of Tweets', 'is_bot': 'Account Type'},
        color_discrete_map={'Human': colors['human'], 'Bot': colors['bot']}
    )

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font={'color': colors['text']},
        xaxis_title="Date",
        yaxis_title="Number of Tweets",
        legend_title="Account Type"
    )

    return fig

@app.callback(
    [Output('bot-wordcloud', 'src'),
     Output('human-wordcloud', 'src')],
    [Input('interval-component', 'n_intervals')]
)
def update_wordclouds(n):
    """Update the word clouds for bot and human tweets."""
    if 'cleaned_text' not in merged_df.columns:
        # Return empty images if no text data
        return '', ''
    
    # Generate word clouds
    bot_text = ' '.join(merged_df[merged_df['is_bot'] == 1]['cleaned_text'].fillna(''))
    human_text = ' '.join(merged_df[merged_df['is_bot'] == 0]['cleaned_text'].fillna(''))
    
    bot_wordcloud_src = f"data:image/png;base64,{generate_wordcloud_image(bot_text)}"
    human_wordcloud_src = f"data:image/png;base64,{generate_wordcloud_image(human_text)}"
    
    return bot_wordcloud_src, human_wordcloud_src

@app.callback(
    [Output('interval-component', 'disabled'),
     Output('interval-component', 'interval'),
     Output('monitoring-status', 'children')],
    [Input('start-monitoring', 'n_clicks'),
     Input('stop-monitoring', 'n_clicks')],
    [State('refresh-interval', 'value'),
     State('interval-component', 'disabled')]
)
def toggle_monitoring(start_clicks, stop_clicks, refresh_interval, is_disabled):
    ctx = getattr(dash, 'ctx', None) or dash.callback_context
    if hasattr(ctx, 'triggered_id'):
        button_id = ctx.triggered_id
        if button_id is None:
            return True, 60*1000, html.P("Monitoring is stopped", style={'color': 'red'})
    else:
        if not ctx.triggered:
            return True, 60*1000, html.P("Monitoring is stopped", style={'color': 'red'})
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'start-monitoring':
        return False, refresh_interval*1000, html.P(f"Monitoring active - Refreshing every {refresh_interval} seconds", style={'color': 'green'})
    return True, refresh_interval*1000, html.P("Monitoring is stopped", style={'color': 'red'})

# Run the app
if __name__ == '__main__':
    print("Starting Twitter Bot Detection Dashboard...")
    print(f"Data loaded: {len(merged_df)} records")
    app.run(debug=False, use_reloader=False, host='127.0.0.1', port=8050)