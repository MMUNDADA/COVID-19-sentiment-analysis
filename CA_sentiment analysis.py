#!/usr/bin/env python
# coding: utf-8

# In[93]:


import pandas as pd
import numpy as np
import pandas as pd
from configparser import ConfigParser
import numpy as np
import spacy
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


# In[92]:


import sys
get_ipython().system('{sys.executable} -m pip install yellowbrick')


# In[2]:


tweetsDF = pd.read_csv("covid_tweets_cleaned_Group6.csv")


# In[3]:


tweetsDF


# ### Cleaning dataframe

# In[5]:


us_state_abbrev = {
    'alabama': 'AL',
    'alaska': 'AK',
    'american samoa': 'AS',
    'arizona': 'AZ',
    'arkansas': 'AR',
    'california': 'CA',
    'colorado': 'CO',
    'connecticut': 'CT',
    'delaware': 'DE',
    'district of columbia': 'DC',
    'florida': 'FL',
    'georgia': 'GA',
    'guam': 'GU',
    'hawaii': 'HI',
    'idaho': 'ID',
    'illinois': 'IL',
    'indiana': 'IN',
    'iowa': 'IA',
    'kansas': 'KS',
    'kentucky': 'KY',
    'louisiana': 'LA',
    'maine': 'ME',
    'maryland': 'MD',
    'massachusetts': 'MA',
    'michigan': 'MI',
    'minnesota': 'MN',
    'mississippi': 'MS',
    'missouri': 'MO',
    'montana': 'MT',
    'nebraska': 'NE',
    'nevada': 'NV',
    'new hampshire': 'NH',
    'new jersey': 'NJ',
    'new mexico': 'NM',
    'new york': 'NY',
    'north carolina': 'NC',
    'north dakota': 'ND',
    'northern mariana islands':'MP',
    'ohio': 'OH',
    'oklahoma': 'OK',
    'oregon': 'OR',
    'pennsylvania': 'PA',
    'palau' : 'PW',
    'puerto rico': 'PR',
    'rhode island': 'RI',
    'south carolina': 'SC',
    'south dakota': 'SD',
    'tennessee': 'TN',
    'texas': 'TX',
    'utah': 'UT',
    'vermont': 'VT',
    'virgin islands': 'VI',
    'virginia': 'VA',
    'washington': 'WA',
    'west virginia': 'WV',
    'wisconsin': 'WI',
    'wyoming': 'WY'
}


# In[12]:


tweetsDF['state'] = tweetsDF['Location'].map(us_state_abbrev)
tweetsDF.state.fillna(tweetsDF.Location, inplace=True)
tweetsDF['state'] = tweetsDF['state'].str.upper()


# In[22]:


tweetsDF['Processed_Tweet'] = tweetsDF['Processed_Tweet'].replace(r'http\S+', "", regex=True).replace(r'www\S+', "", regex=True)


# In[24]:


tweetsDF


# ### Extracting CA tweets

# In[25]:


is_CA =  tweetsDF['state']=='CA'
CA_DF = tweetsDF[is_CA]


# In[26]:


CA_DF


# In[27]:


import en_core_web_sm
nlp = en_core_web_sm.load()


# ### Removing stop words

# In[41]:


CA_DF["Processed_Tweet"] = CA_DF["Processed_Tweet"].astype(str)


# In[43]:


def pre_pipe(doc_tweets):
    tweet_lemm = ' '.join(token.lemma_ for token in doc_tweets if not (token.is_stop or token.lemma_ == '-PRON-'))
    t_pos = ','.join(token.pos_ for token in doc_tweets if not (token.is_stop or token.lemma_ == '-PRON-'))
    return tweet_lemm, t_pos


# In[44]:


def processing_pipe(raw_tweets):
    covid_tweets=[]
    pos=[]
    t_num = 1
    # it's performant to use 'nlp.pipe' since it allows for multi-threaded batch processing of individual tweets or 'docs'
    for doc in nlp.pipe(raw_tweets, batch_size=7500, disable=["parser"]): # batch_size dependent on system CPU/RAM
        covid_tweets.append(pre_pipe(doc)[0])
        pos.append(pre_pipe(doc)[1])
        t_num += 1
    return covid_tweets, pos


# In[47]:


t_clean = processing_pipe(CA_DF['Processed_Tweet'])

CA_DF['Tweet'] = pd.Series(t_clean[0][:])
CA_DF['POS'] = pd.Series(t_clean[1][:])

print('Tweets processed, POS added, removing original tweet column...')

try:
    CA_DF['Processed_Tweet'].replace('', np.nan, inplace=True) 
    CA_DF=CA_DF.dropna()
except:
    raise
print('Tweets removed.')


# In[49]:


CA_DF['Tweet'] = pd.Series(t_clean[0][:])
CA_DF['POS'] = pd.Series(t_clean[1][:])

print('Tweets processed, POS added, removing original tweet column...')

try:
    CA_DF['Processed_Tweet'].replace('', np.nan, inplace=True) 
    CA_DF=CA_DF.dropna()
except:
    raise
print('Tweets removed.')


# In[58]:


df_c = CA_DF['Tweet'].to_list()
r_tweet = " ".join(df_c)
print('Tweets appended to single string.')
# Create a basic wordcloud
print(f'Generating WordCloud across {len(r_tweet)} characters.')
wordcloud = WordCloud(background_color="white", collocations=False, max_words=100).generate(r_tweet)
print('WordCloud generated and displaying...')
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[60]:


nlp = spacy.load('en_core_web_sm')


# In[61]:


t_clean


# In[70]:


from pandas import DataFrame
CA = DataFrame(df_c,columns=['Tweet'])


# In[71]:


total_tweets = len(CA.index)


# In[74]:


CA = CA.replace('[^A-Za-z0-9 ]+','', regex=True)


# ### Build a word count line graph to explore density of the terms

# In[75]:


print('Pulling out indivdual words.')
all_words = []
for line in CA['Tweet']:
    words = line.split()
    for word in words:
        all_words.append(word.lower())
print('Words extracted.')
print(all_words[:40])


# In[76]:


word_count = Counter(all_words).most_common(1000)
word_count_x = []
word_count_y = []
for word, count in word_count:
    word_count_x.append(word)
    word_count_y.append(count)


# In[81]:


plt.figure(figsize=(30,10))
plt.plot(word_count_x[850:], word_count_y[850:], linestyle='-', linewidth=1) # play around with the values within x and y to slice the data differently
plt.ylabel("Count")
plt.xlabel("Word")
plt.xticks(rotation=90)
plt.title('Plot of words frequency in corpus (for California)')
plt.show()


# ### TDM and TF-IDF

# In[84]:


#Using - bi-frams and min occurance as 10

vect = CountVectorizer(ngram_range=(2,2), min_df=10, stop_words=['fuck', 'shit', 'ass'])
tfidf_vect = TfidfVectorizer(ngram_range=(2,2), min_df=10, stop_words=['fuck', 'shit', 'ass'])


# In[85]:


#Fitting across out tweets
tdm_data = vect.fit_transform(CA['Tweet'])
tfidf_data = tfidf_vect.fit_transform(CA['Tweet'])
tdm_columns = vect.get_feature_names()
tfidf_columns = tfidf_vect.get_feature_names()


# In[90]:


#printing matices stats
print(f'Number of TDM features: {len(tdm_columns)}') 
print(f'Number of TF-IDF features: {len(tfidf_columns)}')
print(f'TDM data: {tdm_data[:3288].toarray()}')
print(f'TDM data: {tfidf_data[:3288].toarray()}')


# ### K-means

# In[115]:


#Further cleaning the tweets
def clean_text(text):
    #remove RT @user and QT @user
    cleaned = re.sub(r'RT @([A-Za-z]+[A-Za-z0-9-_]+):','',text)
    cleaned = re.sub(r'QT @([A-Za-z]+[A-Za-z0-9-_]+):','',cleaned)
    #remove @user
    cleaned = re.sub(r'@([A-Za-z]+[A-Za-z0-9-_]+)','',cleaned)

    #remove multiple spaces
    cleaned = re.sub('\s+', ' ', cleaned)
    
    #removing single alphabets
    cleaned =  re.sub(r'\b\w\b','',cleaned)

    #make everything lowercase
    lower_text = cleaned.lower()

    #separate hypenated words
    separate = lower_text.split('-')
    combined = ' '.join(separate)

    #strip beginning and ending whitespace
    clean_spaces = combined.strip()
    
  

    return clean_spaces

def clean_emojis(string):
    return string.encode('ascii', 'ignore').decode('ascii')


# In[116]:


CA['Clean_Tweet'] = CA['Tweet'].apply(clean_text)


# In[117]:


CA


# In[119]:


#increasing the minimum occurances to 30

ca_tweets = CA['Clean_Tweet'].tolist()
vectorizer = TfidfVectorizer(ngram_range=(2,2), min_df=30)
X = vectorizer.fit_transform(ca_tweets)


# In[120]:


print('Running K-mean elbow diagram')
model = KMeans()
visualizer = KElbowVisualizer(model, k=(4,12))
visualizer.fit(X)
visualizer.show()


# In[121]:


# Appropriate number of clusters in this case is 8
Number_Clusters=8
kmeans = KMeans(n_clusters=Number_Clusters, random_state=0).fit(X)


# In[125]:


Cluster_Labels=kmeans.labels_.tolist()


# In[126]:


data={'Tweet':ca_tweets,'Cluster_Number':Cluster_Labels}
df_10=pd.DataFrame(data)


# In[127]:


df_10


# In[128]:


Actual_Clusters_Ordered=[[n  for n in range(len(Cluster_Labels)) if(Cluster_Labels[n]==i)]
                         for i in range(Number_Clusters)]


# In[131]:


for i in range(len(Actual_Clusters_Ordered)):
    for n in range(len(Actual_Clusters_Ordered[i])):
        String="Cluster_Name_"+str(i)
        print(ca_tweets[Actual_Clusters_Ordered[i][n]],String)


# In[134]:


from sklearn.decomposition import TruncatedSVD # to work with sparse matrices
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances


# In[135]:


sk_tsvd = TruncatedSVD(n_components= 2)
y_tsvd = sk_tsvd.fit_transform(X)
svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
svd.fit(X)


# In[136]:


kmeans.fit(y_tsvd)


# In[137]:


svd_clusters = kmeans.predict(y_tsvd)


# In[138]:


svd_clusters


# In[139]:


plt.scatter(y_tsvd[:, 0], y_tsvd[:, 1], c=svd_clusters, s=50, cmap='magma')
plt.show()


# In[ ]:




