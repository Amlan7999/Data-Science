#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('disaster_tweets_data.csv')
df.head()


# In[3]:


df.info()


# In[4]:


#_null values check
df.isnull().sum()


# In[5]:


#_removes pattern in the input text
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt


# In[6]:


#_remove twitter handles 
df['processed_tweet_data'] = np.vectorize(remove_pattern)(df['tweets'], "#[\w]*")
df.head()


# In[7]:


#_remove special characters, numbers and punctuations
df['processed_tweet_data'] = df['processed_tweet_data'].str.replace("[^a-zA-Z#]", " ")
df.head()


# In[8]:


#_convert words to lower case
df['processed_tweet_data'].str.lower()


# In[9]:


#_individual words considered as tokens
tokenized_tweet = df['processed_tweet_data'].apply(lambda x: x.split())
tokenized_tweet.head()


# In[10]:


import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")


# In[11]:


stopwords_list = stopwords.words("english")


# In[12]:


#_remove stop words
def stopword_clean(x):
    return ",".join([word for word in str(x).split() if word not in stopwords_list])
df['processed_tweet_data'] = df['processed_tweet_data'].apply(lambda x : stopword_clean(x))
df.head()


# In[13]:


#_stem the words
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
tokenized_tweet.head()


# In[14]:


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = " ".join(tokenized_tweet[i])
    
df['processed_tweet_data'] = tokenized_tweet
df.head()


# In[15]:


get_ipython().system('pip install wordcLoud')


# In[16]:


#_visualize the frequent words
all_words =" ".join([sentence for sentence in df['processed_tweet_data']])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500 , random_state=42, max_font_size=100).generate(all_words)

#plot the figure
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[17]:


#_visualize the (-ve) frequent words
all_words =" ".join([sentence for sentence in df['processed_tweet_data'][df['target']==1]])

wordcloud = WordCloud(width=800, height=500 , random_state=42, max_font_size=100).generate(all_words)

#plot the figure
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[18]:


#_visualize the (+ve) frequent words
all_words =" ".join([sentence for sentence in df['processed_tweet_data'][df['target']==0]])

wordcloud = WordCloud(width=800, height=500 , random_state=42, max_font_size=100).generate(all_words)

#plot the figure
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[19]:


# feature extraction
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(df['processed_tweet_data'])


# In[20]:


bow[0].toarray()


# In[21]:


x = df['processed_tweet_data']


# In[22]:


x.shape


# In[23]:


y = df['target'] 
y.shape


# In[24]:


#TRAIN AND TEST ML MODELS


# In[44]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(bow, df['target'], random_state=42, test_size=0.25)


# In[45]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score


# In[46]:


# training
model = LogisticRegression()
model.fit(x_train, y_train)


# In[49]:


# testing
pred = model.predict(x_test)
f1_score(y_test, pred)


# In[50]:


accuracy_score(y_test,pred)


# In[51]:


# use probability to get output
pred_prob = model.predict_proba(x_test)
pred = pred_prob[:, 1] >= 0.3
pred = pred.astype(np.int)

f1_score(y_test, pred)


# In[52]:


accuracy_score(y_test,pred)


# In[53]:


pred_prob[0][1] >= 0.3


# In[56]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)


# In[58]:


y_pred = knn.predict(x_test)
y_pred


# In[59]:


Knn = accuracy_score(y_test,y_pred)
print('KNN Classifier Accuracy Score: ',Knn)
cm_rfc=my_confusion_matrix(y_test, y_pred, 'KNN Confusion Matrix')


# In[60]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(x_train, y_train)


# In[62]:


ypred_NB = NB_classifier.predict(x_test)
ypred_NB


# In[63]:


NB = accuracy_score(y_test,ypred_NB)
print(' Multinomial Naïve Bayes Accuracy Score: ',NB)
cm_rfc=my_confusion_matrix(y_test, ypred_NB, ' Multinomial Naïve Bayes Confusion Matrix')


# In[ ]:


#conclusion

LOGISTIC REGRESSION Accuracy :  78.62394957983193
KNN Classifier Accuracy :  70.85084033613446
Multinomial Naïve Bayes Accuracy :  78.72899159663865


# In[ ]:




