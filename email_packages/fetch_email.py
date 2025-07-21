import os
import base64
import re
import pandas as pd
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# ✅ Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# ✅ Function to authenticate Gmail
def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                r'credentials/credentials_email.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

# ✅ Function to remove HTML tags using regex
def remove_html_tags(text):
    if not text:
        return ''
    text = re.sub(r'<[^>]+>', '', text)   # remove all tags
    text = re.sub(r'\s+', ' ', text)      # clean up multiple spaces
    return text.strip()

# ✅ Function to extract email body
def get_email_body(payload):
    def decode_data(data):
        return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')

    parts = payload.get('parts')
    if parts:
        # Prefer text/plain first
        for part in parts:
            if part.get('mimeType') == 'text/plain':
                data = part['body'].get('data')
                if data:
                    return decode_data(data)
        # Fallback to text/html
        for part in parts:
            if part.get('mimeType') == 'text/html':
                data = part['body'].get('data')
                if data:
                    return decode_data(data)
    # Direct body
    body = payload.get('body', {}).get('data')
    if body:
        return decode_data(body)
    return '(No body found)'

# ✅ Fetch emails and clean them
def get_emails(service, max_results=10):
    messages = service.users().messages().list(userId='me', maxResults=max_results).execute().get('messages', [])

    email_list = []
    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        headers = msg_data['payload']['headers']
        payload = msg_data['payload']

        email_info = {'From': '', 'Subject': '', 'Date': '', 'Body': '', 'clean_body': ''}

        for header in headers:
            name = header.get('name')
            if name == 'From':
                email_info['From'] = header.get('value')
            elif name == 'Subject':
                email_info['Subject'] = header.get('value')
            elif name == 'Date':
                email_info['Date'] = header.get('value')

        raw_body = get_email_body(payload)
        email_info['Body'] = raw_body
        email_info['clean_body'] = remove_html_tags(raw_body)

        email_list.append(email_info)

    print(f"✅ Fetched {len(email_list)} emails.")
    return email_list

# ✅ Save to CSV
def save_emails_to_csv(email_list, filename=r'data\emails_cleaned.csv'):
    df = pd.DataFrame(email_list)
    df.to_csv(filename, index=False)
    print(f"✅ Saved cleaned emails to {filename}")

# # ✅ Main
# if __name__ == '__main__':
#     service = authenticate_gmail()
#     emails = get_emails(service, max_results=1)
#     save_emails_to_csv(emails)
