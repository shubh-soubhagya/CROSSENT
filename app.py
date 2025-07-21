
############# TICKET SENTIMENT SEGMENTATION ################

from email.fetch_email import authenticate_gmail, get_emails, save_emails_to_csv
from email.email_agents import extract_name_email, clean_email_body, classify_sentiment
import pandas as pd

####### FETCH EMAIL ################

service = authenticate_gmail()
emails = get_emails(service, max_results=1)
save_emails_to_csv(emails)

############# EMAIL AGENT #################

df = pd.read_csv(r"data\emails_cleaned.csv")
df[['src_name', 'src_email']] = df['From'].apply(extract_name_email)
df['new_body'] = df['Body'].apply(clean_email_body)
df[['emotion_sentiment', 'fine_grained_sentiment', 'thinking']] = df['new_body'].apply(classify_sentiment)
df.to_csv(r"data\emails_cleaned.csv", index=False)
print(f"✅ Completed. Saved {len(df)} emails to emails_cleaned.csv with sentiment analysis.")
