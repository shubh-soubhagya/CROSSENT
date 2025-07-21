import pandas as pd
import re
import os
from dotenv import load_dotenv
from groq import Groq
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentiments import emotion_based_sentiments, fine_grained_sentiments

# ✅ Load API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# ✅ Extract name/email from 'From' column
def extract_name_email(from_text):
    match = re.match(r'(?:"?([^"]*)"?\s)?<?([\w\.-]+@[\w\.-]+)>?', str(from_text))
    if match:
        name = match.group(1) if match.group(1) else ""
        email = match.group(2) if match.group(2) else ""
        return pd.Series([name.strip(), email.strip()])
    return pd.Series(["", ""])

# ✅ Clean the email body using Groq
def clean_email_body(text):
    prompt = f"""
You are a helpful email cleaner. Clean the following email body:
- Remove unnecessary HTML tags and useless formatting.
- Do not change or alter the wording.
- Keep the natural reading flow.
Email Body:
{text}
"""
    chat_completion = client.chat.completions.create(
        model="compound-beta-mini",
        messages=[
            {"role": "system", "content": "You clean email content without changing meaning or wording."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2048
    )
    return chat_completion.choices[0].message.content.strip()

# ✅ Sentiment Classification Agent
def classify_sentiment(text):
    prompt = f"""
You are a sentiment analysis expert.
Given the following email body, classify its sentiment into:
- One of these fine-grained sentiments: {fine_grained_sentiments}
- One of these emotion-based sentiments: {emotion_based_sentiments}
- Provide a brief reasoning (thinking).

Respond ONLY in this format:
Fine-Grained Sentiment: <fine-grained sentiment>
Emotion Sentiment: <emotion-based sentiment>
Thinking: <brief reasoning>

Email Body:
{text}
"""
    chat_completion = client.chat.completions.create(
        model="compound-beta-mini",
        messages=[
            {"role": "system", "content": "You accurately classify sentiments with clear reasoning."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1024
    )
    response = chat_completion.choices[0].message.content.strip()

    # ✅ Extract data using regex
    fine_match = re.search(r"Fine-Grained Sentiment:\s*(.+)", response)
    emotion_match = re.search(r"Emotion Sentiment:\s*(.+)", response)
    thinking_match = re.search(r"Thinking:\s*(.+)", response)

    fine_sentiment = fine_match.group(1).strip() if fine_match else "Unknown"
    emotion_sentiment = emotion_match.group(1).strip() if emotion_match else "Unknown"
    thinking = thinking_match.group(1).strip() if thinking_match else "No reasoning provided."

    return pd.Series([emotion_sentiment, fine_sentiment, thinking])

# # ✅ Load Data
# df = pd.read_csv("emails_cleaned.csv")

# # ✅ Extract name/email
# df[['src_name', 'src_email']] = df['From'].apply(extract_name_email)

# # ✅ Clean body
# df['new_body'] = df['Body'].apply(clean_email_body)

# # ✅ Sentiment Classification
# df[['emotion_sentiment', 'fine_grained_sentiment', 'thinking']] = df['new_body'].apply(classify_sentiment)

# # ✅ Save the final result
# df.to_csv("emails_cleaned.csv", index=False)
# print(f"✅ Completed. Saved {len(df)} emails to emails_cleaned.csv with sentiment analysis.")
