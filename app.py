from fetch_email import authenticate_gmail, get_emails, save_emails_to_csv
from email_agents import extract_name_email, clean_email_body, classify_sentiment
import pandas as pd

####### FETCH EMAIL ################
service = authenticate_gmail()
emails = get_emails(service, max_results=1)
save_emails_to_csv(emails)

############# EMAIL AGENT #################

df = pd.read_csv("emails_cleaned.csv")

# ✅ Extract name/email
df[['src_name', 'src_email']] = df['From'].apply(extract_name_email)

# ✅ Clean body
df['new_body'] = df['Body'].apply(clean_email_body)

# ✅ Sentiment Classification
df[['emotion_sentiment', 'fine_grained_sentiment', 'thinking']] = df['new_body'].apply(classify_sentiment)

# ✅ Save the final result
df.to_csv("emails_cleaned.csv", index=False)
print(f"✅ Completed. Saved {len(df)} emails to emails_cleaned.csv with sentiment analysis.")
