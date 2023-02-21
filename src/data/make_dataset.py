from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build

def main():
    # Set your API credentials
    credentials = service_account.Credentials.from_service_account_file(
        'credentials.json',
        scopes=['https://www.googleapis.com/auth/youtube.readonly'])

    # Create a YouTube API client
    youtube = build('youtube', 'v3', credentials=credentials)

    # Make a request to the API to get a sample of 10 videos
    response = youtube.videos().list(
        part='snippet',
        chart='mostPopular',
        maxResults=10
    ).execute()

    # Print the video high res thumbnail URL
    for item in response['items']:
        print(item['snippet']['thumbnails']['high']['url'])

if __name__ == '__main__':
    main()
