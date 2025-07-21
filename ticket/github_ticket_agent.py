import pandas as pd
import re
import os
from dotenv import load_dotenv
from groq import Groq
import sys
# import fine_grained_sentiments, emotion_based_sentiments
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentiments import fine_grained_sentiments, emotion_based_sentiments

# ✅ Load API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# ✅ Sentiment Classification Function
def classify_sentiment(text):
    prompt = f"""
You are a helpful assistant classifying GitHub issue descriptions.

Classify into:
- Fine-Grained Sentiment: choose from {fine_grained_sentiments}
- Emotion-Based Sentiment: choose from {emotion_based_sentiments}
- Thinking: short explanation why you assigned these sentiments

Respond STRICTLY in this format:
Fine-Grained Sentiment: <fine sentiment>
Emotion Sentiment: <emotion sentiment>
Thinking: <brief reasoning>

Description:
{text}
"""
    chat_completion = client.chat.completions.create(
        model="compound-beta-mini",
        messages=[
            {"role": "system", "content": "You classify GitHub issue sentiments accurately."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1024
    )
    response = chat_completion.choices[0].message.content.strip()

    # ✅ Extract values using regex
    fine_match = re.search(r"Fine-Grained Sentiment:\s*(.+)", response)
    emotion_match = re.search(r"Emotion Sentiment:\s*(.+)", response)
    thinking_match = re.search(r"Thinking:\s*(.+)", response)

    fine_sentiment = fine_match.group(1).strip() if fine_match else "Unknown"
    emotion_sentiment = emotion_match.group(1).strip() if emotion_match else "Unknown"
    thinking = thinking_match.group(1).strip() if thinking_match else "No reasoning provided."

    return pd.Series([emotion_sentiment, fine_sentiment, thinking])

# # ✅ Load CSV
# df = pd.read_csv(r"data\github_issues.csv")

# # ✅ Apply Sentiment Classification
# df[['emotion_sentiment', 'fine_grained_sentiment', 'thinking']] = df['Description'].apply(classify_sentiment)

# # ✅ Save Final CSV
# df.to_csv(r"data\github_issues_with_sentiment.csv", index=False)
# print(f"✅ Completed. {len(df)} issues saved to github_issues_with_sentiments.csv")
