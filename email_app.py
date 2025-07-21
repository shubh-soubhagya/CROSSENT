from flask import Flask, render_template, jsonify
import pandas as pd
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime

app = Flask(__name__)
app.config['DATA_FOLDER'] = 'data'
app.config['CSV_FILE'] = 'emails_cleaned.csv'

# Ensure data folder exists
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

def get_csv_path():
    """Get the path to our specific CSV file"""
    csv_path = os.path.join(app.config['DATA_FOLDER'], app.config['CSV_FILE'])
    return csv_path if os.path.exists(csv_path) else None

def process_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        
        # Convert date column if exists
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                pass
        
        # Generate visualizations
        img_data = generate_visualizations(df)
        
        return {
            'data': df.to_dict('records'),
            'visualizations': img_data,
            'stats': {
                'total_emails': len(df),
                'sentiment_counts': df['emotion_sentiment'].value_counts().to_dict(),
                'fine_grained_counts': df['fine_grained_sentiment'].value_counts().to_dict(),
                'last_updated': datetime.fromtimestamp(
                    os.path.getmtime(filepath)
                ).strftime('%Y-%m-%d %H:%M:%S'),
                'filename': os.path.basename(filepath)
            }
        }
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return None

def generate_visualizations(df):
    img_data = {}
    
    # Sentiment Distribution Pie Chart
    plt.figure(figsize=(8, 6))
    df['emotion_sentiment'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('Emotion Sentiment Distribution')
    plt.ylabel('')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_data['sentiment_pie'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Fine-Grained Sentiment Bar Chart
    plt.figure(figsize=(10, 6))
    df['fine_grained_sentiment'].value_counts().plot.bar()
    plt.title('Fine-Grained Sentiment Distribution')
    plt.xlabel('Sentiment Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_data['fine_grained_bar'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Sentiment Over Time (if Date column exists)
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        try:
            df.set_index('Date', inplace=True)
            sentiment_counts = df.resample('D')['emotion_sentiment'].value_counts().unstack().fillna(0)
            
            plt.figure(figsize=(12, 6))
            sentiment_counts.plot.area(stacked=True)
            plt.title('Sentiment Trend Over Time')
            plt.xlabel('Date')
            plt.ylabel('Email Count')
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_data['sentiment_trend'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        except Exception as e:
            print(f"Error generating trend chart: {e}")
    
    return img_data

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/analyze')
def analyze_data():
    try:
        csv_path = get_csv_path()
        if not csv_path:
            return jsonify({
                'error': True,
                'message': f"CSV file {app.config['CSV_FILE']} not found",
                'solution': "Please ensure emails_cleaned.csv exists in the data folder"
            }), 404
        
        processed_data = process_csv(csv_path)
        if not processed_data:
            return jsonify({
                'error': True,
                'message': 'Error processing CSV file',
                'solution': 'Check the file format and content'
            }), 500
            
        return jsonify({
            'success': True,
            **processed_data
        })
        
    except Exception as e:
        return jsonify({
            'error': True,
            'message': 'Server error occurred',
            'solution': 'Check server logs',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)