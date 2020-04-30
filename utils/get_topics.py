# Common
import pandas as pd
import numpy as np

# Text processing
import re
from bs4 import BeautifulSoup
import nltk

# Visualization modules
import matplotlib.pyplot as plt

# Model modules
from sklearn.feature_extraction.text import TfidfVectorizer#, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim.test.utils import get_tmpfile
from gensim.test.utils import datapath
from gensim.models.ldamulticore import LdaMulticore

from gensim.models import phrases, word2vec

import pyodbc
from sqlalchemy import create_engine
import urllib

# load custom fuctions
from lipht_lda import df_lda_preprocessing, lda_preprocess_string, df_lda_features, get_lda_topics, lda_predict_df, lda_predict_string, document_to_bow, df_lda_features
from utils.lipht_data import getConnection

# SET SQL parameters
engine = ('LIPHT-VM-01','Akademikernes_MSCRM_addition','sa','Google100123!')



# SET Analysis Parameters
dataset = 'Udbetalingsteam'
initiatedby = 'Initiatedby_AKA'
n_gram = 3
sample_size= 10000
no_words= 5000
no_below= 10 # filter out tokens that appear in less than 15 documents
random_state=1
research_scope = 'Udbetalingteam'
num_topics = 7

# Import data
df_import = pd.read_pickle('data/{}/{}.pkl'.format(initiatedby,dataset))

# Create a copy
df = df_import.copy(deep=True)

print(df.shape)

# Process the data
df_lda_preprocessing(df,'FirstMessage',n_gram)

# Save the processed dataaset
df.to_pickle('data/{}/{}.pkl'.format(initiatedby,dataset))

# LOAD the data
df_scope = pd.read_pickle('data/{}/{}.pkl'.format(initiatedby,dataset))

# Check Sample Size
if sample_size>df_scope.shape[0]:
    sample_size = df_scope.shape[0]
print(df_scope.shape, sample_size)

# Name the test
data_scope_name = initiatedby+'_'+research_scope +'_topics-'+ str(num_topics) +'_Sample-'+str(sample_size) +'_WordCount-'+str(no_words) +'_RandomState-'+str(random_state)+'_dataset-'+ dataset
print(data_scope_name)

# Vectorize the words
# Create dictionary with words from df_scope (the total dataset)
dictionary = Dictionary(documents=df_scope.stemmed_text.values)
print("Found {} words.".format(len(dictionary.values())))

dictionary.filter_extremes(no_below=no_below, keep_n=no_words)

dictionary.compactify()  # Reindexes the remaining words after filtering
print("Left with {} words.".format(len(dictionary.values())))

#Make a BoW for every Besked
document_to_bow(df_scope, dictionary)

# Create a sample of Scope
scope_lda_sample = df_scope.sample(sample_size, random_state=random_state)
scope_lda_sample.shape

scope_lda_sample[['ThreadID','ThreadMessageID','ThreadSubject','FirstMemberMessage','text','tokenized_text','stopwords_removed','lemmatized_text','stemmed_text']].head()

# Find optimal number of topics
top_words = [v for v in dictionary.values()]
top_words = list(set(top_words))
df_scope['OnlyTopWords'] = list(map(lambda doc: [word for word in doc if word in top_words], df_scope['stemmed_text']))
print("No of top words: {} ".format(len(top_words)))

from lipht_lda import remove_not_topwords, ListToString
top_words, _ = remove_not_topwords(scope_lda_sample, df_scope)

print("Found {} words.".format(len(dictionary.values())))
scope_lda_sample['clean_content'] = scope_lda_sample['OnlyTopWords'].apply(ListToString)

tfidf_wordvector = TfidfVectorizer(
                analyzer='word', 
                max_df=0.8, 
                min_df=5, 
#                 stop_words=stopwords.words('danish'),
#                 ngram_range=(1,3)
                ) 

#fit the tfidf_wordvector to clean_content
tfidf_wordvector_maxtrix = tfidf_wordvector.fit_transform(scope_lda_sample.clean_content)
print(tfidf_wordvector_maxtrix.shape)

tfidf_wordvector_2d = tfidf_wordvector_maxtrix.todense()
top_range = 50
increments = 1
distortions = []
K = range(1,top_range,increments)
for k in K:
    kmeanModel = KMeans(n_clusters=k, n_jobs=-1, random_state=0).fit(tfidf_wordvector_2d)
    kmeanModel.fit(tfidf_wordvector_2d)
    distortions.append(sum(np.min(cdist(tfidf_wordvector_2d, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / tfidf_wordvector_2d.shape[0])

# Plot the elbow
title_elbow = 'The Elbow Method: {4}. Showing the optimal k\nSample Size: {0}, Top {1} Words, with increments of {2} from 0 to {3}'.format(sample_size, len(top_words), increments, top_range-1, data_scope_name)
info_text = research_scope +'_topics-'+ str(num_topics) +'_Sample-'+str(sample_size) +'_WordCount-'+str(no_words) +'_RandomState-'+str(random_state)
filename = '{}_The Elbow Method_{}'.format(research_scope, info_text)
plt.figure(figsize=(16, 10))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title(title_elbow)
plt.savefig(fname='data/{}/{}.png'.format(initiatedby,filename),transparent=True)
plt.show()

# LDA Model Training

# We want to maximize the probability of the corpus in the training set.
corpus = scope_lda_sample.bow

print(('LDA Model based on {3} dataset.\n\tSample Size: {0},\n\tTop {1} Words,\n\tNo of Topics {2}'.format(sample_size, len(dictionary.values()), num_topics, data_scope_name)))

LDAmodel_scope = LdaMulticore(corpus=corpus,#mm,
                        id2word=dictionary,
                        num_topics=num_topics,
                        workers=4,
                        chunksize=5000,
                        passes=50,
                        alpha='asymmetric',
                        random_state=random_state)

dictionary.save('data/model/{0}_dictionary.pkl'.format(research_scope))#data_scope_name))
LDAmodel_scope.save('data/model/{0}'.format(research_scope))#data_scope_name))
# pickle the model here and insert in SQL
LDAmodel_scope = LdaMulticore.load('data/model/{0}'.format(research_scope))#data_scope_name))

# Feature vector
df_lda_features(LDAmodel_scope, scope_lda_sample)

# Topic distribution
RequestTopicDistribution = scope_lda_sample['lda_features'].mean()
fig, ax1 = plt.subplots(1,1,figsize=(20,6))
nr_top_bars = 5
title_dist ='{}_Request Topic distributions showing top {} bars of {} topics'.format(research_scope, nr_top_bars, num_topics)
ax1.set_title(title_dist, fontsize=16)

for ax, distribution, color in zip([ax1], [RequestTopicDistribution], ['r']):
    # Individual distribution barplots
    ax.bar(range(len(distribution)), distribution, alpha=0.7)
    rects = ax.patches
    for i in np.argsort(distribution)[-nr_top_bars:]:
        rects[i].set_color(color)
        rects[i].set_alpha(1)

fig.tight_layout(h_pad=3.)
fig.savefig(fname='data/{}/{}'.format(initiatedby,title_dist),transparent=True)

# Inspect topics and words
from lipht_lda import get_topics_and_probability, get_lda_topics

df_topics = get_topics_and_probability(scope_lda_sample, LDAmodel_scope, num_topics, 5)
# Upload topics to sql
df_topics.to_sql(name='topics_{}_{}'.format(dataset,initiatedby),con=engine , schema='input', if_exists='replace', index=False)

# get_lda_topics(scope_lda_sample, LDAmodel_scope, num_topics,20)

# Name the topics
# lda_topic_names = {
#     0:'Ferie og feriepenge',
#     1:'Sendt oplysninger til AKA',
#     2:'Ansættelseskontrakt eller frigørelse',
#     3:'Spørgsmål om dagpenge',
#     4:'Ansøgning om befordring',
#     5:'Ansættelse',
#     6:'Ledighed',
#     7:'Adgang',
#     8:'Noget med tid*',
#     9:'Dagpenge mellem jul og nytår',
#     11:'Fejl ved dagpenge',
#     12:'Spørgsmål til blanket',
#     14:'Ydelseskort',
#     15:'Pension og Efterløn',
#     16:'Dagpenge/Supplerende',
#     17:'Spørgsmål til udfyldelse',
#     19:'Spørgsmål om beskæftigelse'
# }

# Test the model
# document = scope_lda_sample.sample(1) # From sample
document = df_scope.sample(1) # From population
doc_id = document['ThreadMessageID']
unseen_document = document['FirstMemberMessage']
print(doc_id, unseen_document)
# Test function and prediction
print(lda_predict_string(unseen_document, LDAmodel_scope, dictionary))#,lda_topic_names))

# Predict topics on Scope
lda_predict_df(df_scope, LDAmodel_scope, dictionary)#, lda_topic_names)

# Save the data with predictions
df_scope.to_pickle('data/{}_{}_with_prediction.pkl'.format(dataset,initiatedby))
# Slim the df
df_scope_sql = df_scope[['ThreadID','ThreadMessageID','text', 'pred_probability', 'pred_index']]#, 'pred_label']]
df_scope_sql.to_sql(name='{}_{}'.format(dataset,initiatedby),con=engine , schema='input', if_exists='replace', index=False)
