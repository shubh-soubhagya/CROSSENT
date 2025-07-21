import os
import re
import csv
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY_2")
client = Groq(api_key=groq_api_key)

# Sentiment categories (matches your existing implementation)
fine_grained_sentiments = [
    "Positive", "Negative", "Neutral", 
    "Satisfied", "Frustrated", "Urgent",
    "Informational", "Complaint", "Appreciation"
]

emotion_based_sentiments = [
    "Happy", "Angry", "Sad", 
    "Excited", "Confused", "Anxious",
    "Grateful", "Impatient", "Hopeful"
]

class CustomerCareChatbot:
    def __init__(self):
        self.conversation = []
        self.ticket_counter = 1000
        self.model = "llama3-70b-8192"  # Using your preferred model architecture

    def extract_contact_info(self, user_input):
        """Extract name/email from user input (matches your email processing approach)"""
        match = re.match(r'(?:"?([^"]*)"?\s)?<?([\w\.-]+@[\w\.-]+)>?', str(user_input))
        if match:
            name = match.group(1) or ""
            email = match.group(2) or ""
            return name.strip(), email.strip()
        return "", ""

    def clean_user_input(self, text):
        """Clean user input using same approach as your email cleaner"""
        prompt = f"""
        You are a helpful text cleaner. Clean the following:
        - Remove unnecessary formatting
        - Preserve original meaning
        - Keep natural flow
        Text: {text}
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You clean text content without changing meaning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2048
        )
        return response.choices[0].message.content.strip()

    def classify_sentiment(self, text):
        """Identical to your sentiment classification function"""
        prompt = f"""
        Classify this message's sentiment:
        Fine-grained options: {fine_grained_sentiments}
        Emotion options: {emotion_based_sentiments}
        Respond in format:
        Fine-Grained Sentiment: <value>
        Emotion Sentiment: <value>
        Thinking: <reasoning>

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
        """Generate response using your preferred prompt structure"""
        prompt = f"""
        You are a customer care assistant. 
        Last user message: {user_input}
        Respond helpfully and professionally.
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
        """Log conversation with same structure as your email processor"""
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
        """Save conversation in identical format to your email processor"""
        if not self.conversation:
            return

        df = pd.DataFrame(self.conversation)
        filename = f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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

            # Process input using your existing functions
            cleaned_input = self.clean_user_input(user_input)
            sentiment = self.classify_sentiment(cleaned_input)
            response = self.generate_response(cleaned_input)
            
            print(f"\n🤖 Bot: {response}\n")
            print(f"Sentiment: {sentiment[0]} | {sentiment[1]}\n")
            
            self.log_conversation(cleaned_input, response, sentiment)

if __name__ == "__main__":
    chatbot = CustomerCareChatbot()
    chatbot.start_chat()