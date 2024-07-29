import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# Model to perform multi-class text classification
text_clf = pipeline('text-classification', model='bhadresh-savani/bert-base-uncased-emotion', return_all_scores=True)
# Model to perform binary sentiment analysis
sentiment_analysis = pipeline('sentiment-analysis', return_all_scores=True)

def get_sentiment_scores(df):
    # Iterate through the data and predict the emotion scores
    for i, text in tqdm(enumerate(df['combined']), desc='Processing', total=len(df), bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):
        # Prediction dictionary (limit the text to 512 tokens, otherwise the model will throw an error)
        sentiment_pred = sentiment_analysis(text[:512])[0]
        text_clf_pred = text_clf(text[:512])[0]
        # Collecting the scores for each emotion
        positive = sentiment_pred[0]['score']
        sadness = text_clf_pred[0]['score']
        joy = text_clf_pred[1]['score']
        love = text_clf_pred[2]['score']
        anger = text_clf_pred[3]['score']
        fear = text_clf_pred[4]['score']
        surprise = text_clf_pred[5]['score']
        # Assign the scores to the corresponding columns
        df.loc[i, 'positive'] = positive
        df.loc[i, 'sadness'] = sadness
        df.loc[i, 'joy'] = joy
        df.loc[i, 'love'] = love
        df.loc[i, 'anger'] = anger
        df.loc[i, 'fear'] = fear
        df.loc[i, 'surprise'] = surprise
    return df

# Perform sentiment analysis on the data, both for the binary and multi-class classification, both on the title and the text
def get_title_sentiment_scores(df):
    # Iterate through the data and predict the emotion scores
    for i, text in tqdm(enumerate(df['title']), desc='Title processing', total=len(df), bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):
        # Prediction dictionary (limit the text to 512 tokens, otherwise the model will throw an error)
        sentiment_pred = sentiment_analysis(text[:512])[0]
        text_clf_pred = text_clf(text[:512])[0]
        # Collecting the scores for each emotion
        positive = sentiment_pred[0]['score']
        sadness = text_clf_pred[0]['score']
        joy = text_clf_pred[1]['score']
        love = text_clf_pred[2]['score']
        anger = text_clf_pred[3]['score']
        fear = text_clf_pred[4]['score']
        surprise = text_clf_pred[5]['score']
        # Assign the scores to the corresponding columns
        df.loc[i, 'title_positive'] = positive
        df.loc[i, 'title_sadness'] = sadness
        df.loc[i, 'title_joy'] = joy
        df.loc[i, 'title_love'] = love
        df.loc[i, 'title_anger'] = anger
        df.loc[i, 'title_fear'] = fear
        df.loc[i, 'title_surprise'] = surprise
    return df

def get_text_sentiment_scores(df):
    # Iterate through the data and predict the emotion scores
    for i, text in tqdm(enumerate(df['text']), desc='Text processing', total=len(df), bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):
        # Prediction dictionary (limit the text to 512 tokens, otherwise the model will throw an error)
        sentiment_pred = sentiment_analysis(text[:512])[0]
        text_clf_pred = text_clf(text[:512])[0]
        # Collecting the scores for each emotion
        positive = sentiment_pred[0]['score']
        sadness = text_clf_pred[0]['score']
        joy = text_clf_pred[1]['score']
        love = text_clf_pred[2]['score']
        anger = text_clf_pred[3]['score']
        fear = text_clf_pred[4]['score']
        surprise = text_clf_pred[5]['score']
        # Assign the scores to the corresponding columns
        df.loc[i, 'text_positive'] = positive
        df.loc[i, 'text_sadness'] = sadness
        df.loc[i, 'text_joy'] = joy
        df.loc[i, 'text_love'] = love
        df.loc[i, 'text_anger'] = anger
        df.loc[i, 'text_fear'] = fear
        df.loc[i, 'text_surprise'] = surprise
    return df