import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd
import re
import joblib

string = "https://www.youtube.com/watch?v=ddexZhkHSOA"
def key(string):
 list=string.split('=')
 vidkey=list[1]
 return vidkey

vidkey=key(string)

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "YOUR API KEY"  # Replace with your actual API key

youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)


# Initialize variables
comments = []
next_page_token = None

# Use a loop to retrieve all commentsjupyter no
while True:
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=vidkey,
        maxResults=100,
        pageToken=next_page_token
    )
    response = request.execute()

    for item in response['items']:
        comment_data = item['snippet']['topLevelComment']['snippet']
        likes = comment_data.get('likeCount', 0)  # Extract the number of likes
        comment_data['likes'] = likes  # Add a 'likes' key to the dictionary
        comments.append(comment_data)

    # Check if there are more pages
    if 'nextPageToken' in response:
        next_page_token = response['nextPageToken']
    else:
        break


df = pd.DataFrame(comments)

df = pd.DataFrame(data =df,columns = ['textDisplay', 'likeCount'])

df['label'] = df['likeCount'].apply(lambda x: 1 if x > 0 else 0)

def remove_emojis(text):
    # Define the regular expression pattern to match emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emojis in the first range
                               u"\U0001F300-\U0001F5FF"  # Emojis in the second range
                               u"\U0001F680-\U0001F6FF"  # Emojis in the third range
                               u"\U0001F700-\U0001F77F"  # Emojis in the fourth range
                               u"\U0001F780-\U0001F7FF"  # Emojis in the fifth range
                               u"\U0001F800-\U0001F8FF"  # Emojis in the sixth range
                               u"\U0001F900-\U0001F9FF"  # Emojis in the seventh range
                               u"\U0001FA00-\U0001FA6F"  # Emojis in the eighth range
                               u"\U0001FA70-\U0001FAFF"  # Emojis in the ninth range
                               u"\U0001FA00-\U0001FA6F"  # Emojis in the tenth range
                               u"\U0001FAD0-\U0001FAFF"  # Emojis in the eleventh range
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F700-\U0001F77F"
                               u"\U0001F780-\U0001F7FF"
                               u"\U0001F800-\U0001F8FF"
                               u"\U0001F900-\U0001F9FF"
                               u"\U0001FA00-\U0001FA6F"
                               u"\U0001FA70-\U0001FAFF"
                               u"\U0001FA00-\U0001FA6F"
                               u"\U0001FAD0-\U0001FAFF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U0001F004"
                               u"\U0001F0CF"
                               u"\U0001F170-\U0001F251"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F004"
                               u"\U0001F0CF"
                               u"\U0001F004"
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F004"
                               u"\U0001F0CF"
                               u"\U0001F004"
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F004"
                               u"\U0001F0CF"
                               u"\U0001F004"
                               u"\U0001F004"
                               "]+", flags=re.UNICODE)

    # Remove emojis from the text
    text_without_emojis = emoji_pattern.sub(r'', text)
    return text_without_emojis

# Example usage:
# Assuming you have a DataFrame 'df' and a column 'text' that contains text with emojis
df['textDisplay'] = df['textDisplay'].apply(remove_emojis)

import string
def nopunct(mess):
    nopunc = [x for x in mess if x not in string.punctuation]
    nopunc = ''.join(nopunc)
    return nopunc

df['textDisplay'] = df['textDisplay'].apply(nopunct)

def brtags(mess):
    mess = mess.replace('<br>','')
    return mess

df['textDisplay'] = df['textDisplay'].apply(brtags)

from sklearn.feature_extraction.text import CountVectorizer
# Vectorize the text data using CountVectorizer
bow_transformer = CountVectorizer(analyzer='word').fit(df['textDisplay'])
reviews_bow = bow_transformer.transform(df['textDisplay'])


from sklearn.feature_extraction.text import TfidfTransformer
# Transform the bag-of-words into TF-IDF features
tfidf_transformer = TfidfTransformer().fit(reviews_bow)
reviews_tfidf = tfidf_transformer.transform(reviews_bow)

# Save the CountVectorizer (bow_transformer) to a .pkl file
bow_transformer_filename = 'bow_transformer.pkl'
joblib.dump(bow_transformer, bow_transformer_filename)

# Save the TfidfTransformer (tfidf_transformer) to a .pkl file
tfidf_transformer_filename = 'tfidf_transformer.pkl'
joblib.dump(tfidf_transformer, tfidf_transformer_filename)


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(reviews_tfidf, df['label'])


# Save the trained model to a .pkl file
model_filename = 'sentiment_model.pkl'
joblib.dump(model, model_filename)



