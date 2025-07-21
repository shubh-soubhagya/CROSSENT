import os
import re
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

# ✅ Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# ✅ Sentiment categories
emotion_based_sentiments = [
    "Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"
]

fine_grained_sentiments = [
    "Very Positive", "Positive", "Neutral", "Negative", "Very Negative"
]

class CustomerCareChatbot:
    def __init__(self):
        self.conversation = []
        self.ticket_counter = 1000
        self.model = "compound-beta-mini"

    def extract_contact_info(self, user_input):
        match = re.match(r'(?:"?([^"]*)"?\s)?<?([\w\.-]+@[\w\.-]+)>?', str(user_input))
        if match:
            name = match.group(1) or ""
            email = match.group(2) or ""
            return name.strip(), email.strip()
        return "", ""

    def classify_sentiment(self, text):
        prompt = f"""
Classify this message's sentiment:
Fine-grained options: {fine_grained_sentiments}
Emotion options: {emotion_based_sentiments}
Respond in format:
Fine-Grained Sentiment: <value>
Emotion Sentiment: <value>
Thinking: <reasoning>

Only give one word value.

Message: {text}
"""
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You accurately classify sentiments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        result = response.choices[0].message.content.strip()

        fine_match = re.search(r"Fine-Grained Sentiment:\s*(.+)", result)
        emotion_match = re.search(r"Emotion Sentiment:\s*(.+)", result)
        thinking_match = re.search(r"Thinking:\s*(.+)", result)

        return (
            emotion_match.group(1).strip() if emotion_match else "Unknown",
            fine_match.group(1).strip() if fine_match else "Unknown",
            thinking_match.group(1).strip() if thinking_match else ""
        )

    def generate_response(self, user_input):
        prompt = f"""
You are a customer care assistant.
Last user message: {user_input}
Respond helpfully and professionally in 15 words only.
"""
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You resolve customer issues professionally."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()

    def log_conversation(self, user_input, bot_response, sentiment_data):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conversation.append({
            "timestamp": timestamp,
            "user_input": user_input,
            "bot_response": bot_response,
            "emotion_sentiment": sentiment_data[0],
            "fine_grained_sentiment": sentiment_data[1],
            "thinking": sentiment_data[2],
            "model_used": self.model
        })

    def save_to_csv(self):
        if not self.conversation:
            return
        df = pd.DataFrame(self.conversation)
        filename = r"data\chatlogs.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Saved {len(df)} interactions to {filename}")

    def start_chat(self):
        print(f"\n🤖 Customer Care Bot (Model: {self.model})")
        print("Type 'quit' to exit\n")

        while True:
            user_input = input("👤 You: ").strip()
            if user_input.lower() == 'quit':
                self.save_to_csv()
                print("🤖 Session ended. Conversation saved.")
                break

            sentiment = self.classify_sentiment(user_input)
            response = self.generate_response(user_input)

            print(f"\n🤖 Bot: {response}\n")
            print(f"Sentiment: {sentiment[0]} | {sentiment[1]}\n")

            self.log_conversation(user_input, response, sentiment)

if __name__ == "__main__":
    chatbot = CustomerCareChatbot()
    chatbot.start_chat()
