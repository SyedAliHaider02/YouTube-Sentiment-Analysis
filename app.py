from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import joblib
import string
import googleapiclient.discovery
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = 'b83a1e0ea4e74d22c5d6a3a0ff5e6e66'

class MyForm(FlaskForm):
    url = StringField('Enter the URL', validators=[DataRequired()])
    comment = StringField('Comment: ', validators=[DataRequired()])
    submit = SubmitField('Submit')

# Load the trained model
model = joblib.load('sentiment_model.pkl')
bow_transformer = joblib.load('bow_transformer.pkl')
tfidf_transformer = joblib.load('tfidf_transformer.pkl')

def key(url):
    list = url.split('=')
    vidkey = list[1]
    return vidkey

@app.route('/', methods=['GET', 'POST'])
def index():
    form = MyForm(request.form)
    accuracy_score = None
    prediction = None
    comments_with_likes = None
    
    if form.validate_on_submit():
        url = form.url.data
        comment = form.comment.data
        accuracy_score, prediction, comments_with_likes= dataset(url, comment)
    return render_template('front.html', form=form, accuracy_score=accuracy_score, prediction=prediction,comments_with_likes=comments_with_likes)




def dataset(url,comment):
    vidkey = key(url)
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyDWCrNAYPqjieCCzC6Un2jEsmYlPkELJlI"  # Replace with your actual API key

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
     
     
     df2 = df.sort_values(by='likeCount', ascending=False)
     
     ten = []
     likes=[]
     if not df2.empty:
      for i in range(0, min(10, len(df2))):
        ten.append(df2['textDisplay'].iloc[i])
        likes.append(df2['likeCount'].iloc[i])

     comments_with_likes = list(zip(ten, likes))

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

    
        text_without_emojis = emoji_pattern.sub(r'', text)
        return text_without_emojis


    df['textDisplay'] = df['textDisplay'].apply(remove_emojis)

       
    def nopunct(mess):
         nopunc = [x for x in mess if x not in string.punctuation]
         nopunc = ''.join(nopunc)
         return nopunc

    df['textDisplay'] = df['textDisplay'].apply(nopunct)

    def brtags(mess):
        mess = mess.replace('<br>','')
        return mess

    df['textDisplay'] = df['textDisplay'].apply(brtags)


    bow_transformer = CountVectorizer(analyzer='word').fit(df['textDisplay'])
    reviews_bow = bow_transformer.transform(df['textDisplay'])



    tfidf_transformer = TfidfTransformer().fit(reviews_bow)
    reviews_tfidf = tfidf_transformer.transform(reviews_bow)

    
    modell = MultinomialNB().fit(reviews_tfidf, df['label'])

    all_predictions = modell.predict(reviews_tfidf)
    prediction = modell.predict(bow_transformer.transform([comment]))

    x=accuracy_score(df['label'], all_predictions)
    x=x*100
    # ...
    return x, prediction,comments_with_likes # Return the accuracy score

if __name__ == '__main__':
    app.run(debug=True)
