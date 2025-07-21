# app.py (updated)
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import os
from datetime import datetime
from email_packages.fetch_email import authenticate_gmail, get_emails, save_emails_to_csv
from email_packages.email_agents import extract_name_email, clean_email_body, classify_sentiment
from chat import CustomerCareChatbot
import requests

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Initialize chatbot (but don't start it yet)
chatbot = CustomerCareChatbot()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/emails')
def email_dashboard():
    try:
        # Fetch and process emails
        service = authenticate_gmail()
        emails = get_emails(service, max_results=10)
        save_emails_to_csv(emails)
        
        df = pd.read_csv(r"data\emails_cleaned.csv")
        if 'From' in df.columns:
            df[['src_name', 'src_email']] = df['From'].apply(extract_name_email)
        if 'Body' in df.columns:
            df['new_body'] = df['Body'].apply(clean_email_body)
        if 'new_body' in df.columns:
            df[['emotion_sentiment', 'fine_grained_sentiment', 'thinking']] = df['new_body'].apply(classify_sentiment)
        df.to_csv(r"data\emails_cleaned.csv", index=False)
        
        email_data = pd.read_csv(r"data\emails_cleaned.csv").to_dict('records')
        return render_template('email_dashboard.html', emails=email_data)
    except Exception as e:
        return render_template('email_dashboard.html', error=str(e))

@app.route('/tickets')
def ticket_dashboard():
    try:
        # Fetch GitHub issues
        repo = "microsoft/vscode"
        per_page = 10
        url = f"https://api.github.com/repos/{repo}/issues?state=all&per_page={per_page}"
        response = requests.get(url)

        if response.status_code == 200:
            issues = response.json()
            issues = [issue for issue in issues if 'pull_request' not in issue]

            df = pd.DataFrame([
                {
                    'Issue ID': issue['id'],
                    'Title': issue['title'],
                    'Description': issue.get('body', ''),
                    'Created At': issue['created_at'],
                    'State': issue['state'],
                    'Issue URL': issue['html_url']
                }
                for issue in issues
            ])

            df[['emotion_sentiment', 'fine_grained_sentiment', 'thinking']] = df['Description'].apply(classify_sentiment)
            df.to_csv(r"data\github_issues_with_sentiment.csv", index=False)
        
        ticket_data = pd.read_csv(r"data\github_issues_with_sentiment.csv").to_dict('records')
        return render_template('ticket_dashboard.html', tickets=ticket_data)
    except Exception as e:
        return render_template('ticket_dashboard.html', error=str(e))

@app.route('/chatbot', methods=['GET', 'POST'])
def chat_interface():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if user_input:
            bot_response = chatbot.generate_response(user_input)
            
            # Classify sentiment of the conversation
            sentiment = classify_sentiment(user_input + " " + bot_response)
            
            # Save to chat log
            chat_log = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'user_input': user_input,
                'bot_response': bot_response,
                'emotion_sentiment': sentiment[0],
                'fine_grained_sentiment': sentiment[1],
                'thinking': sentiment[2],
                'model_used': 'Groq'  # Assuming you're using Groq based on your imports
            }
            
            # Append to CSV
            chat_df = pd.DataFrame([chat_log])
            if not os.path.exists(r"data\chat_logs.csv"):
                chat_df.to_csv(r"data\chat_logs.csv", index=False)
            else:
                chat_df.to_csv(r"data\chat_logs.csv", mode='a', header=False, index=False)
            
            return jsonify({
                'bot_response': bot_response,
                'sentiment': sentiment
            })
    
    return render_template('chat_interface.html')

@app.route('/chatlogs')
def chat_dashboard():
    try:
        if os.path.exists(r"data\chat_logs.csv"):
            chat_data = pd.read_csv(r"data\chat_logs.csv").to_dict('records')
        else:
            chat_data = []
        return render_template('chat_dashboard.html', chats=chat_data)
    except Exception as e:
        return render_template('chat_dashboard.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)