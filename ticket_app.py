############### TICKET SYSTEM ###############

import requests
import pandas as pd

# ✅ Configure repository and number of issues
repo = "microsoft/vscode"  # Example repo (you can change this)
per_page = 10  # Number of issues to fetch

# ✅ GitHub API endpoint
url = f"https://api.github.com/repos/{repo}/issues?state=all&per_page={per_page}"

response = requests.get(url)

if response.status_code == 200:
    issues = response.json()
    # ✅ Filter out Pull Requests (issues only)
    issues = [issue for issue in issues if 'pull_request' not in issue]

    # ✅ Create DataFrame with selected fields
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

    df.to_csv(r"data\github_issues.csv", index=False)
    print(f"✅ Saved {len(df)} issues to github_issues.csv")
else:
    print(f"❌ Failed to fetch issues: {response.status_code} - {response.text}")

import pandas as pd
import re
import os
from dotenv import load_dotenv
from groq import Groq

df_ticket = pd.read_csv(r"data\github_issues.csv")

# ✅ Apply Sentiment Classification
from email_packages.email_agents import classify_sentiment
df_ticket[['emotion_sentiment', 'fine_grained_sentiment', 'thinking']] = df_ticket['Description'].apply(classify_sentiment)

# ✅ Save Final CSV
df_ticket.to_csv(r"data\github_issues_with_sentiment.csv", index=False)
print(f"✅ Completed. {len(df_ticket)} issues saved to github_issues_with_sentiments.csv")