import numpy as np
import pandas as pd

pd.options.display.max_colwidth = 100
train_data = pd.read_csv("../input/train.csv", encoding='ISO-8859-1')

rand_indexs = np.random.randint(1,len(train_data),50).tolist()
train_data["SentimentText"][rand_indexs]
tweets_text = train_data.SentimentText.str.cat()
emos = set(re.findall(r" ([xX:;][-']?.) ",tweets_text))
emos_count = []
for emo in emos:
emos_count.append((tweets_text.count(emo), emo))
sorted(emos_count,reverse=True)

HAPPY_EMO = r" ([xX;:]-?[dD)]|:-?[\)]|[;:][pP]) "
SAD_EMO = r" (:'?[/|\(]) "
print("Happy emoticons:", set(re.findall(HAPPY_EMO, tweets_text)))
print("Sad emoticons:", set(re.findall(SAD_EMO, tweets_text)))

import nltk
from nltk.tokenize import word_tokenize

def most_used_words(text):
tokens = word_tokenize(text)
frequency_dist = nltk.FreqDist(tokens)
print("There is %d different words" % len(set(tokens)))
return sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)

most_used_words(train_data.SentimentText.str.cat())[:100]

from nltk.corpus import stopwords


mw = most_used_words(train_data.SentimentText.str.cat())
most_words = []
for w in mw:
if len(most_words) == 1000:
break
if w in stopwords.words("english"):
continue
else:
most_words.append(w)

sorted(most_words)

from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

def stem_tokenize(text):
stemmer = SnowballStemmer("english")
stemmer = WordNetLemmatizer()
return [stemmer.lemmatize(token) for token in word_tokenize(text)]

def lemmatize_tokenize(text):
lemmatizer = WordNetLemmatizer()
return [lemmatizer.lemmatize(token) for token in word_tokenize(text)]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline

class TextPreProc(BaseEstimator,TransformerMixin):
def __init__(self, use_mention=False):
self.use_mention = use_mention
  
def fit(self, X, y=None):
return self
  
def transform(self, X, y=None):

if self.use_mention:
X = X.str.replace(r"@[a-zA-Z0-9_]* ", " @tags ")
else:
X = X.str.replace(r"@[a-zA-Z0-9_]* ", "")
  
X = X.str.replace("#", "")
X = X.str.replace(r"[-\.\n]", "")
X = X.str.replace(r"&\w+;", "")
X = X.str.replace(r"https?://\S*", "")
X = X.str.replace(r"(.)\1+", r"\1\1")
X = X.str.replace(HAPPY_EMO, " happyemoticons ")
X = X.str.replace(SAD_EMO, " sademoticons ")
X = X.str.lower()
return X

from sklearn.model_selection import train_test_split

sentiments = train_data['Sentiment']
tweets = train_data['SentimentText']

vectorizer = TfidfVectorizer(tokenizer=lemmatize_tokenize, ngram_range=(1,2))
pipeline = Pipeline([
('text_pre_processing', TextPreProc(use_mention=True)),
('vectorizer', vectorizer),
])

learn_data, test_data, sentiments_learning, sentiments_test = train_test_split(tweets, sentiments, test_size=0.3)

learning_data = pipeline.fit_transform(learn_data)


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

lr = LogisticRegression()
bnb = BernoulliNB()
mnb = MultinomialNB()

models = {
'logitic regression': lr,
'bernoulliNB': bnb,
'multinomialNB': mnb,
}

for model in models.keys():
scores = cross_val_score(models[model], learning_data, sentiments_learning, scoring="f1", cv=10)
print("===", model, "===")
print("scores = ", scores)
print("mean = ", scores.mean())
print("variance = ", scores.var())
models[model].fit(learning_data, sentiments_learning)
print("score on the learning data (accuracy) = ", accuracy_score(models[model].predict(learning_data), sentiments_learning))
print("")


from sklearn.model_selection import GridSearchCV

grid_search_pipeline = Pipeline([
('text_pre_processing', TextPreProc()),
('vectorizer', TfidfVectorizer()),
('model', MultinomialNB()),
])

params = [
{
'text_pre_processing__use_mention': [True, False],
'vectorizer__max_features': [1000, 2000, 5000, 10000, 20000, None],
'vectorizer__ngram_range': [(1,1), (1,2)],
},
]
grid_search = GridSearchCV(grid_search_pipeline, params, cv=5, scoring='f1')
grid_search.fit(learn_data, sentiments_learning)
print(grid_search.best_params_)

mnb.fit(learning_data, sentiments_learning)
testing_data = pipeline.transform(test_data)
mnb.score(testing_data, sentiments_test)

sub_data = pd.read_csv("../input/test.csv", encoding='ISO-8859-1')
sub_learning = pipeline.transform(sub_data.SentimentText)
sub = pd.DataFrame(sub_data.ItemID, columns=("ItemID", "Sentiment"))
sub["Sentiment"] = mnb.predict(sub_learning)
print(sub)
model = MultinomialNB()
model.fit(learning_data, sentiments_learning)
tweet = pd.Series([input(),])
tweet = pipeline.transform(tweet)
proba = model.predict_proba(tweet)[0]
print("The probability that this tweet is sad is:", proba[0])
print("The probability that this tweet is happy is:", proba[1])

