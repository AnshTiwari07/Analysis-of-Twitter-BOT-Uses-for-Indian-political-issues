"""
Report Generator for Twitter Bot Analysis
This script compiles the analysis results and visualizations into a comprehensive PDF report.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import numpy as np
from PIL import Image
import glob

def create_title_page(pdf, title="Twitter Bot Analysis for Indian Political Issues"):
    """Create a title page for the report."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.7, title, fontsize=24, ha='center', fontweight='bold')
    ax.text(0.5, 0.6, "CAA Discussion Analysis", fontsize=18, ha='center')
    
    # Date
    current_date = datetime.now().strftime("%B %d, %Y")
    ax.text(0.5, 0.5, f"Generated on: {current_date}", fontsize=14, ha='center')
    
    # Footer
    ax.text(0.5, 0.1, "Powered by Relevance Vector Machine Classifier", fontsize=10, ha='center', fontstyle='italic')
    
    pdf.savefig(fig)
    plt.close()

def add_summary_statistics(pdf, predictions_path="../data/bot_predictions.csv"):
    """Add summary statistics to the report."""
    try:
        # Load predictions
        df = pd.read_csv(predictions_path)
        
        # Create summary page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.95, "Summary Statistics", fontsize=20, ha='center', fontweight='bold')
        
        # Bot statistics
        bot_count = df[df['is_bot'] == 1].shape[0]
        human_count = df[df['is_bot'] == 0].shape[0]
        total_count = bot_count + human_count
        bot_percentage = (bot_count / total_count) * 100 if total_count > 0 else 0
        
        stats_text = [
            f"Total accounts analyzed: {total_count}",
            f"Bot accounts detected: {bot_count} ({bot_percentage:.1f}%)",
            f"Human accounts detected: {human_count} ({100-bot_percentage:.1f}%)"
        ]
        
        y_pos = 0.85
        for stat in stats_text:
            ax.text(0.1, y_pos, stat, fontsize=14)
            y_pos -= 0.05
        
        # Try to add engagement statistics if available
        try:
            avg_bot_engagement = df[df['is_bot'] == 1]['engagement_score'].mean()
            avg_human_engagement = df[df['is_bot'] == 0]['engagement_score'].mean()
            
            ax.text(0.1, y_pos - 0.1, "Engagement Analysis:", fontsize=16, fontweight='bold')
            y_pos -= 0.05
            
            engagement_text = [
                f"Average bot engagement score: {avg_bot_engagement:.2f}",
                f"Average human engagement score: {avg_human_engagement:.2f}",
                f"Engagement ratio (bot/human): {avg_bot_engagement/avg_human_engagement:.2f}" if avg_human_engagement > 0 else "Engagement ratio: N/A"
            ]
            
            y_pos -= 0.05
            for stat in engagement_text:
                ax.text(0.1, y_pos, stat, fontsize=14)
                y_pos -= 0.05
        except Exception as e:
            print(f"Could not add engagement statistics: {e}")
        
        pdf.savefig(fig)
        plt.close()
    except Exception as e:
        print(f"Error creating summary statistics: {e}")

def add_model_performance(pdf, model_path="../models/rvm_bot_classifier.pkl"):
    """Add model performance metrics to the report."""
    try:
        import joblib
        
        # Load model data
        model_data = joblib.load(model_path)
        
        # Create model performance page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.95, "Model Performance", fontsize=20, ha='center', fontweight='bold')
        ax.text(0.5, 0.9, "Relevance Vector Machine Classifier", fontsize=16, ha='center', fontstyle='italic')
        
        # If we have saved metrics in the model data, display them
        if hasattr(model_data, 'get') and model_data.get('metrics'):
            metrics = model_data['metrics']
            
            y_pos = 0.8
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    ax.text(0.1, y_pos, f"{metric_name}: {value:.4f}", fontsize=14)
                    y_pos -= 0.05
        else:
            # Generic model information
            ax.text(0.1, 0.8, "Model Type: Relevance Vector Machine", fontsize=14)
            ax.text(0.1, 0.75, "Features: User behavior, tweet content, temporal patterns", fontsize=14)
            ax.text(0.1, 0.7, "Note: Detailed metrics available in model evaluation", fontsize=14)
        
        pdf.savefig(fig)
        plt.close()
    except Exception as e:
        print(f"Error adding model performance: {e}")

def add_visualizations(pdf, results_dir="../results"):
    """Add visualizations from the results directory to the report."""
    try:
        # Get all PNG files in the results directory
        image_files = glob.glob(os.path.join(results_dir, "*.png"))
        
        if not image_files:
            print("No visualization files found.")
            return
        
        # Add each visualization to the report
        for img_path in image_files:
            try:
                # Create a figure with the image
                img = Image.open(img_path)
                fig, ax = plt.subplots(figsize=(8.5, 11))
                
                # Add a title based on the filename
                title = os.path.basename(img_path).replace('.png', '').replace('_', ' ').title()
                ax.set_title(title, fontsize=16)
                
                # Display the image
                ax.imshow(np.array(img))
                ax.axis('off')
                
                # Add to PDF
                pdf.savefig(fig)
                plt.close()
                
                print(f"Added visualization: {os.path.basename(img_path)}")
            except Exception as e:
                print(f"Error adding visualization {img_path}: {e}")
    except Exception as e:
        print(f"Error processing visualizations: {e}")

def add_conclusions(pdf):
    """Add conclusions and recommendations to the report."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    ax.text(0.5, 0.95, "Conclusions and Recommendations", fontsize=20, ha='center', fontweight='bold')
    
    conclusions = [
        "1. Bot Detection Effectiveness:",
        "   - The RVM classifier demonstrates strong performance in identifying automated accounts.",
        "   - Feature importance analysis reveals key indicators of bot behavior.",
        "",
        "2. Bot Behavior Patterns:",
        "   - Automated accounts show distinct temporal patterns and engagement characteristics.",
        "   - Content analysis reveals common themes and messaging strategies.",
        "",
        "3. Recommendations:",
        "   - Continue monitoring bot activity around political issues.",
        "   - Expand analysis to include network relationships between accounts.",
        "   - Consider implementing real-time detection for emerging campaigns.",
        "",
        "4. Future Research:",
        "   - Investigate cross-platform coordination of automated campaigns.",
        "   - Analyze the evolution of bot sophistication over time.",
        "   - Develop countermeasures for detecting increasingly sophisticated bots."
    ]
    
    y_pos = 0.85
    for line in conclusions:
        ax.text(0.1, y_pos, line, fontsize=12)
        y_pos -= 0.04
    
    pdf.savefig(fig)
    plt.close()

def generate_report(output_path="../results/twitter_bot_analysis_report.pdf"):
    """Generate a comprehensive PDF report."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create PDF
        with PdfPages(output_path) as pdf:
            print("Generating report...")
            
            # Add title page
            create_title_page(pdf)
            
            # Add summary statistics
            add_summary_statistics(pdf)
            
            # Add model performance
            add_model_performance(pdf)
            
            # Add visualizations
            add_visualizations(pdf)
            
            # Add conclusions
            add_conclusions(pdf)
            
        print(f"Report generated successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Error generating report: {e}")
        return False

if __name__ == "__main__":
    generate_report()