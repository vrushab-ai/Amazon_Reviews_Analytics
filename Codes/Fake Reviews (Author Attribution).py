
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from collections import Counter
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

import re

# NLTK Libraries
import nltk
from nltk.corpus import stopwords
from nltk import TweetTokenizer
from nltk.util import skipgrams
from nltk import FreqDist

from itertools import chain
from scipy.cluster import hierarchy
from sklearn.model_selection import train_test_split

import seaborn as sns
from scipy.stats import norm




# Dropping the rows where the pandas is failing to read the data.
data_electronics = pd.read_csv('dataset/amazon_reviews_us_Electronics_v1_00.tsv', sep='\t', error_bad_lines=False)
data_electronics.dropna(inplace=True)


# In[18]:


data = data_electronics.copy()


# #### Datasets with #minReviews of 50 per customer

# In[19]:


data = data[data.groupby('customer_id')['review_id'].transform('count')>=50]
data.reset_index(drop=True, inplace=True)


# In[20]:


data.head(2)


# In[21]:


data['review_body'].apply(lambda x: len(x.split(" "))).describe()


# <b>Interpretation:</b>
# <br>
# There are reviews with just 1 word also with maximum number of words goes till 4981

# #### Datasets with minimum 100 words

# In[22]:


data_100 = data[data['review_body'].apply(lambda x: len(x.split(" ")))>=100]
data_100.reset_index(drop=True, inplace=True)


# #### Keeping only relevant features

# In[57]:


df = data[['customer_id','review_body']]
df_100 = data_100[['customer_id','review_body']]


# ### Data Pre-processing

# In[59]:


def most_common_stop_words(df):
    df['tokenize'] = df['review_body'].map(tokenize)
    df['stop_words'] = df['tokenize'].map(stop_words_filter)
    # Finding the most common stop words in the reviews
    stop_words_corpus = []
    for i in df['stop_words']:
        stop_words_corpus.extend(i)

    common_stop_words = dict(FreqDist(stop_words_corpus).most_common(50)).keys()
    
    return common_stop_words

def tokenize(s):
    s = s.lower()
    token = TweetTokenizer()
    return token.tokenize(s)

def pos_tagger(s):
    return [i[1] for i in nltk.pos_tag(s)]

stops = stopwords.words('english')
x = [i.split("'")for i in stops]
stops = [i[0] for i in x]
stops = list(set(stops))
slang_stops = ['gonna', 'coulda', 'shoulda',
               'lotta', 'lots', 'oughta', 'gotta', 'ain', 'sorta', 'kinda', 'yeah', 'whatever', 'cuz', 'ya', 'haha', 'lol', 'eh']
puncts = ['!', ':', '...', '.', '%', '$', "'", '"', ';']
formattings = ['##', '__', '_', '    ', '*', '**']
stops.extend(slang_stops)
stops.extend(puncts)
stops.extend(formattings)
# stops.extend(skips)

def stop_words_filter(s):
    return [i for i in s if i in stops]

def filter_most_common_stop_words(s):
    return [i for i in s if i in common_stop_words]

def skip_grams(s):
    grams = []
    for i in skipgrams(s, 2, 2):
        grams.append(str(i))
    return grams

def tf(review_list):
    l = []
    j = 0
    all_words = defaultdict(list)
    for i in review_list:
        
        freq = Counter(i)
        freq_dict = dict(zip( freq.keys(), (np.array(list(freq.values())) / len(i)) ))
        l.append(freq_dict)
        for k,m in freq_dict.items():
            all_words[k].append(m)
    return l,all_words

def normalize(review_list, all_words):
    l = []
    for i in review_list:
        temp = copy.deepcopy(i)
        for k,m in i.items():
            if (np.std(all_words[k])) == 0:
                temp.pop(k, None)
            else:
                temp[k] = (temp[k] - np.mean(all_words[k])) / (np.std(all_words[k]))
        l.append(temp)
    return l


# In[60]:


def data_processing(df1):
    df1= df1.groupby('customer_id').apply(lambda x: " ".join(x['review_body']))
    df1 = df1.reset_index().rename(columns={0: 'reviews'})
    
    #Tokenizing the reviews
    df1['tokenize'] = df1['reviews'].map(tokenize)
    
    #POS Tagging
    df1['pos_tag'] = df1['tokenize'].map(pos_tagger)
    
    # Finding Stop words
    df1['stop_words'] = df1['tokenize'].map(stop_words_filter)
    # Note to self: Consider only Coordinate conjuctions, preposition as a function words instead øf all words

    # Filtering the most common stop words
    df1['stop_words'] = df1['stop_words'].map(filter_most_common_stop_words)
    
    # POS Tagging Skip Grams
    df1['skip_grams'] = df1['pos_tag'].map(skip_grams)
    
    # Concatenating stop_words and skip_grams
    df1['features'] = [df1.ix[i, 'stop_words'] + df1.ix[i, 'skip_grams'] for i in range(len(df1))]
    
    review_list, all_words = tf(df1['features'])
    normalized_data = normalize(review_list, all_words)
    df1['scaled'] = normalized_data
    first = pd.DataFrame(normalized_data).fillna(0)
    
    return first, df1


# #### Modelling using complete dataset




# Determining most common stop words
common_stop_words = most_common_stop_words(df)

### Spliting the data into two anonymized datasets
df1, df2 = train_test_split(df, test_size=0.5, stratify= df['customer_id'] )
print(f"Df1 Shape: {df1.shape}\nDf2 Shape: {df2.shape}\n")

# Mapping Pre-processing functions
first, df1 = data_processing(df1)
second, df2 = data_processing(df2)

# Selecting only common features
cols = set(first.columns) & set(second.columns)
results = cosine_similarity(first[cols], second[cols])
results_df = pd.DataFrame({'actual': list(df1.index), 'pred': np.argmax(results, axis=1), 'value': np.max(results, axis=1)})
print(f"Results: {results_df.head()}\n")





similarity_df = pd.DataFrame(results).unstack().reset_index()
similarity_df.columns = ['users_I', 'users_II', 'cosine_similarity']
similarity_df['match_flag'] = [1 if (similarity_df.iloc[i, 0] == similarity_df.iloc[i, 1]) else 0 for i in range(len(similarity_df))]
matches = similarity_df[similarity_df['match_flag']==1]
non_matches = similarity_df[similarity_df['match_flag']==0]
sns.distplot(non_matches['cosine_similarity'], label='non-match')
sns.distplot(matches['cosine_similarity'], label = 'match')
plt.legend()





alpha = 0.001

# define probability
p = 1 - alpha

# retrieve value <= probability
value = norm.ppf(p)
print(f"Critical Z-Value: {np.round(value,2)}")

# confirm with cdf
p = norm.cdf(value)
print(f"Probability: {p}")





non_matches['z_score'] = (non_matches['cosine_similarity'] - np.mean(non_matches['cosine_similarity'])) / (np.std(non_matches['cosine_similarity']))
non_match_temp = non_matches[non_matches['z_score']>=(3.0)].sort_values(by='z_score')
print(non_match_temp.head())
print("\nThreshold value of Cosine Similarity: ", np.round(non_match_temp.iloc[0]['cosine_similarity'], 2))


# ### Modelling on data with 100 words in each reviews




# Determining most common stop words
common_stop_words = most_common_stop_words(df_100)

### Spliting the data into two anonymized datasets
df1_100, df2_100 = train_test_split(df_100, test_size=0.5, stratify= df_100['customer_id'] )
print(f"Df1 Shape: {df1_100.shape}\nDf2 Shape: {df2_100.shape}\n")

# Mapping Pre-processing functions
first_100, df1_100 = data_processing(df1_100)
second_100, df2_100 = data_processing(df2_100)

# Selecting only common features
cols_100 = set(first_100.columns) & set(second_100.columns)
results_100 = cosine_similarity(first_100[cols_100], second_100[cols_100])
results_df_100 = pd.DataFrame({'actual': list(df1_100.index), 'pred': np.argmax(results_100, axis=1), 'value': np.max(results_100, axis=1)})
print(f"Results: {results_df_100.head()}\n")





similarity_df_100 = pd.DataFrame(results_100).unstack().reset_index()
similarity_df_100.columns = ['users_I', 'users_II', 'cosine_similarity']
similarity_df_100['match_flag'] = [1 if (similarity_df_100.iloc[i, 0] == similarity_df_100.iloc[i, 1]) else 0 for i in range(len(similarity_df_100))]
matches_100 = similarity_df_100[similarity_df_100['match_flag']==1]
non_matches_100 = similarity_df_100[similarity_df_100['match_flag']==0]
sns.distplot(non_matches_100['cosine_similarity'], label='non-match')
sns.distplot(matches_100['cosine_similarity'], label = 'match')
plt.legend()





alpha = 0.001

# define probability
p = 1 - alpha

# retrieve value <= probability
value = norm.ppf(p)
print(f"Critical Z-Value: {np.round(value,2)}")

# confirm with cdf
p = norm.cdf(value)
print(f"Probability: {p}")



non_matches_100['z_score'] = (non_matches_100['cosine_similarity'] - np.mean(non_matches_100['cosine_similarity'])) / (np.std(non_matches_100['cosine_similarity']))
non_match_temp_100 = non_matches_100[non_matches_100['z_score']>=(3.0)].sort_values(by='z_score')
print(non_match_temp_100.head())
print("\nThreshold value of Cosine Similarity: ", np.round(non_match_temp_100.iloc[0]['cosine_similarity'], 2))


# <b>Interpretation:</b><br>
# By comparing both charts, we realize the impact of number of words in an comment. If we do analyze with more comments, we can analyse customer's writing style better.
# <br>
# After selection of threshold,let's do clustering on whole data and decide the cluster size based on our threshold.



all_dataset, final_df = data_processing(df)


import scipy.cluster.hierarchy as shc
plt.figure(figsize=(15, 8))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(all_dataset, method='single', metric='cosine'),  color_threshold=0.65)
plt.axhline(y=0.65, color='r', linestyle='--')


from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=len(results) - 29, affinity='cosine', linkage='single')  
cluster.fit_predict(all_dataset)



final_df['cluster_labels'] = cluster.labels_
fake_users_clusters = final_df['cluster_labels'].value_counts()[final_df['cluster_labels'].value_counts()>1].index
fake_users_clusters



fake_users = final_df[['customer_id', 'cluster_labels']]
fake_users['fake'] = fake_users['cluster_labels'].apply(lambda x: 1 if x in fake_users_clusters else 0)
fake_users = fake_users[fake_users['fake']==1]


# ### Storing fake customer ids



fake_users.to_csv('same_authors.csv')

