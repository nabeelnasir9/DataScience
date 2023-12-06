import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import Counter
plt.style.use('ggplot')
import re
import string
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('stopwords')
import os
from wordcloud import wordcloud

data = pd.read_csv('sentiment-analysis.csv')
print("Original Data:")
data.head()
print(data.head())

# data=data['Text, Sentiment, Source, Date/Time, User ID, Location, Confidence Score'].str.split(',', expand=True)
data[['Text', 'Sentiment', 'Source', 'Date/Time', 'User ID', 'Location', 'Confidence Score']] = data['Text, Sentiment, Source, Date/Time, User ID, Location, Confidence Score'].str.split(',', expand=True)
data.drop(columns=['Text, Sentiment, Source, Date/Time, User ID, Location, Confidence Score'], inplace=True)
#data.columns=['Text', 'Sentiment', 'Source', 'Date/Time', 'User ID', 'Location', 'Confidence Score']
print("\nData after splitting:")
print(data.head())

print('There are {} rows and {} columns in data.'.format(data.shape[0], data.shape[1]))
data.head(10)
data.isnull().sum()
data.dropna(inplace=True)
data.head(10)
data['Date/Time'] = data['Date/Time'].str.strip()

data[['Date', 'Time']] = data['Date/Time'].str.split(' ', expand=True)

data.drop(columns=['Date/Time'], inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time
sns.countplot(x='Sentiment', data=data, palette= ['green', 'red'])
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Count of Positive and Negative Reviews')
plt.show()
data['Sentiment'] = data['Sentiment'].str.strip()
data['Sentiment'] = data['Sentiment'].str.lower()


fig, axes = plt.subplots(2, 2, figsize=(12, 8))

review_length_negative = data[data['Sentiment'] == 'negative']['Text'].str.len()
axes[0, 0].hist(review_length_negative, color='red')
axes[0, 0].set_title("Negative Sentiment - Characters")

review_length_positive = data[data['Sentiment'] == 'positive']['Text'].str.len()
axes[0, 1].hist(review_length_positive, color='green')
axes[0, 1].set_title("Positive Sentiment - Characters")

review_length_negative_words = data[data['Sentiment'] == 'negative']['Text'].str.split().map(lambda x: len(x))
axes[1, 0].hist(review_length_negative_words, color='red')
axes[1, 0].set_title("Negative Sentiment - Words")

review_length_positive_words = data[data['Sentiment'] == 'positive']['Text'].str.split().map(lambda x: len(x))
axes[1, 1].hist(review_length_positive_words, color='green')
axes[1, 1].set_title("Positive Sentiment - Words")

plt.tight_layout()

plt.show()
data['DayOfWeek'] = data['Date'].dt.day_name()
custom_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
data['DayOfWeek'] = pd.Categorical(data['DayOfWeek'], categories=custom_order, ordered=True)

sns.countplot(x='DayOfWeek', hue='Sentiment', data=data, palette= ['green', 'red'])

plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.title('Sentiment Distribution by Weekday')
plt.legend(title='Sentiment', labels=['Positive', 'Negative'], bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()
sns.countplot(y='Source', hue='Sentiment', data=data, palette=['green', 'red'])
plt.xlabel('Count')
plt.ylabel('Source')
plt.title('Positive and Negative Reviews by Source')
plt.legend(title='Sentiment', loc='center left', bbox_to_anchor=(1, 0.5), labels=['Positive', 'Negative'])
plt.show()
sns.countplot(y='Location', hue='Sentiment', data=data, palette=['green', 'red'])
plt.xlabel('Count')
plt.ylabel('Source')
plt.title('Positive and Negative Reviews by Location')
plt.legend(title='Sentiment', loc='center left', bbox_to_anchor=(1, 0.5), labels=['Positive', 'Negative'])
plt.show()
data.describe()
data['User ID'] = data['User ID'].str.strip()

user_reviews = data.groupby('User ID').agg(
    ReviewCount=('Text', 'count'),
    SentimentDistribution=('Sentiment', lambda x: dict(x.value_counts()))
).reset_index()

user_reviews = user_reviews[user_reviews['ReviewCount'] >= 2]

sentiment_data = pd.DataFrame(user_reviews['SentimentDistribution'].to_list())
sentiment_data.index = user_reviews['User ID']
sentiment_data
data[data['User ID'] == 'user456'][['Text', 'Source']]
def create_corpus(sentiment):
    corpus=[]
    for x in data[data['Sentiment']==sentiment]['Text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus
def count_punctuation_marks(sentiment, punctuation):
    count = 0
    for text in data[data['Sentiment'] == sentiment]['Text']:
        count += text.count(punctuation)
    return count

def plot_punctuation_counts():
    punctuations = ['!', '.']
    sentiments = ['positive', 'negative']
    colors = ['green', 'red']

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for idx, punctuation in enumerate(punctuations):
        for sentiment, color in zip(sentiments, colors):
            count = count_punctuation_marks(sentiment, punctuation)
            axes[idx].bar(sentiment, count, color=color)
        
        axes[idx].set_xlabel('Sentiment')
        axes[idx].set_ylabel('Count')
        axes[idx].set_title(f'Number of "{punctuation}" in reviews')
        axes[idx].legend(sentiments)

    plt.tight_layout()

plot_punctuation_counts()
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

corpus_negative = create_corpus('negative')
dic_negative = defaultdict(int)
for word in corpus_negative:
    if word in stop:
        dic_negative[word] += 1
top_negative = sorted(dic_negative.items(), key=lambda x: x[1], reverse=True)[:10]
x_neg, y_neg = zip(*top_negative)
axes[0].bar(x_neg, y_neg, color='red')
axes[0].set_xlabel('Words')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Top 10 Most Common Stop Words (Negative Sentiment)')

corpus_positive = create_corpus('positive')
dic_positive = defaultdict(int)
for word in corpus_positive:
    if word in stop:
        dic_positive[word] += 1
top_positive = sorted(dic_positive.items(), key=lambda x: x[1], reverse=True)[:10]
x_pos, y_pos = zip(*top_positive)
axes[1].bar(x_pos, y_pos, color='green')
axes[1].set_xlabel('Words')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Top 10 Most Common Stop Words (Positive Sentiment)')

plt.tight_layout()
plt.show()
data['Text'] = data['Text'].str.strip('"')
def plot_most_common_words(corpus, sentiment, color):
    data['Text'] = data['Text'].str.replace(r'[{}]'.format(string.punctuation), '')
    word_freq = defaultdict(int)
    special_chars = string.punctuation
    for word in corpus:
        word = word.lower()  
        if word not in stop and word not in special_chars:
            word_freq[word] += 1

    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    x, y = zip(*top_words)

    plt.barh(x, y, color=color)  
    plt.ylabel('Words')          
    plt.xlabel('Frequency')      
    plt.title(f'Most Common Words ({sentiment.capitalize()} Sentiment)')

corpus_positive = create_corpus('positive')
plt.subplot(1, 2, 1)
plot_most_common_words(corpus_positive, 'positive', 'green')

corpus_negative = create_corpus('negative')
plt.subplot(1, 2, 2)
plot_most_common_words(corpus_negative, 'negative', 'red')
plt.subplots_adjust(wspace=0.5)  

plt.gcf().set_size_inches(12, 6)
plt.tight_layout()
plt.show()
from wordcloud import WordCloud, STOPWORDS

def plot_word_cloud(sentiment):
    corpus = create_corpus(sentiment)
    stopwords = set(STOPWORDS)
    special_chars = set(string.punctuation)
    custom_stopwords = set(['would', 'could', 'should', 'might', 'will', 'can', 'must']) 

    all_stopwords = stopwords.union(special_chars).union(custom_stopwords)

    wordcloud = WordCloud(width=800, height=400, stopwords=all_stopwords, background_color='white').generate(' '.join(corpus))

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment.capitalize()} Sentiment')
    plt.show()

plot_word_cloud('negative')
plot_word_cloud('positive')



