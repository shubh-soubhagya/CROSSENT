import pandas as pd
import re
import os
from dotenv import load_dotenv
from groq import Groq
from sentiments import emotion_based_sentiments, fine_grained_sentiments

# ✅ Load API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# ✅ Sentiment Classification Agent
def classify_ticket_sentiment(text):
    prompt = f"""
Classify the sentiment of the following customer ticket description.

Provide:
- Fine-Grained Sentiment from: {fine_grained_sentiments}
- Emotion-Based Sentiment from: {emotion_based_sentiments}

Respond strictly in this format:
Fine-Grained Sentiment: <fine-grained sentiment>
Emotion Sentiment: <emotion-based sentiment>

Ticket Description:
{text}
"""
    chat_completion = client.chat.completions.create(
        model="compound-beta-mini",
        messages=[
            {"role": "system", "content": "You classify sentiments accurately for support tickets."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=512
    )
    response = chat_completion.choices[0].message.content.strip()

    fine_match = re.search(r"Fine-Grained Sentiment:\s*(.+)", response)
    emotion_match = re.search(r"Emotion Sentiment:\s*(.+)", response)

    fine_sentiment = fine_match.group(1).strip() if fine_match else "Unknown"
    emotion_sentiment = emotion_match.group(1).strip() if emotion_match else "Unknown"

    return pd.Series([emotion_sentiment, fine_sentiment])

# ✅ Load Data
df = pd.read_csv("balanced_tickets.csv")

# ✅ Apply Sentiment Classification
df[['emotion_sentiment', 'fine_grained_sentiment']] = df['Ticket Description'].apply(classify_ticket_sentiment)

# ✅ Save Final Result
df.to_csv("balanced_tickets.csv", index=False)
print(f"✅ Completed. Saved {len(df)} tickets to 'tickets_with_sentiments.csv'.")
