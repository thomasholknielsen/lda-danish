def TrainLDAModel(corpus, dictionary, lda_num_topics, workers, lda_chunksize, lda_passes, lda_alpha='asymmetric', lda_eta=None, lda_minimum_probability=0.01, random_state=0, log=None):
    from gensim.models.ldamulticore import LdaMulticore
    from datetime import datetime
    
    start = datetime.now()
    trained_model = LdaMulticore(corpus=corpus,#mm,
                            id2word=dictionary,
                            num_topics=lda_num_topics,
                            workers=4,
                            chunksize=lda_chunksize,
                            passes=lda_passes,
                            alpha=lda_alpha,
                            eta=lda_eta,
                            minimum_probability=lda_minimum_probability,
                            random_state=random_state)
    end = datetime.now()
    if log:
        log['train_start'] = start
        log['train_end'] = end

    return trained_model


def PredictTopicFromBOW(bow_vector, lda_model, lda_topic_name_dict=None, prediction_index=None):
    "Input a string prepared bow-vector"
    best_prediction = 0
    if prediction_index is None:
        for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
            if score > best_prediction:
                prediction_index = index
                best_prediction = score

        if lda_topic_name_dict is None:
            pred = [best_prediction, prediction_index]#, lda_model.print_topic(index, 5)
        else:
            pred = [best_prediction, prediction_index, lda_topic_name_dict[prediction_index]]#, lda_model.print_topic(index, 5)
        return pred
    else:
        for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
            if index == prediction_index:
                best_prediction = score
                pred = [best_prediction, prediction_index]
                
                return pred



def lda_predict_df(df, col_name, lda_model, dictionary, lda_topic_name_dict=None, only_best_prediction=True):
    """ Make it possible to clean the df if that hasnt happend yet
        It should be possible to select the dataframe to be predicted
        
        The current function assumes that data is clean and has a column named stemmed_text"""
#     for index, score in sorted(LDAmodel_lang[bow_vector], key=lambda tup: -1*tup[1]):
#         print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
    cols = list(df.columns)
    df['bow'] = list(map(lambda doc: dictionary.doc2bow(doc), df[col_name]))
    if only_best_prediction:
        if lda_topic_name_dict is None:
            df['prediction'] = df['bow'].apply(PredictTopicFromBOW,lda_model=lda_model)
            df[['pred_probability','pred_index']] = pd.DataFrame(df.prediction.values.tolist(), index= df.index)
        else:
            df['prediction'] = df['bow'].apply(PredictTopicFromBOW,lda_model=lda_model, lda_topic_name_dict=lda_topic_name_dict)
            df[['pred_probability','pred_index','pred_label']] = pd.DataFrame(df.prediction.values.tolist(), index= df.index)
        df.drop(['prediction'], axis=1)
    else:
        num_topics = len(lda_model.get_topics())
        for i in range(num_topics):
            df[i] = df['bow'].apply(PredictTopicFromBOW,lda_model=lda_model, prediction_index=i)
#         df[['pred_probability','pred_index']] = pd.DataFrame(df.prediction.values.tolist(), index= df.index)

        # Unpivot values, and split predictions
        values = [i for i in range(num_topics)]
        df = pd.melt(df, id_vars=cols, value_vars=values)
        df = df[df['value'].isnull()==False].sort_values(by=[col_name])
        df.rename(columns={'variable':'index','value':'prediction'}, inplace=True)
        df[['pred_probability','pred_index']] = pd.DataFrame(df.prediction.values.tolist(), index=df.index)
    
    return df
 

# Above this line is necessary




import nltk
import re
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from utils.lipht_data import df_clean_sting, df_simple_clean_string
import pandas as pd
import numpy as np







def df_make_features_from_string(df, col_name):
    """ Add 
        - Count of Characters       -> int
        - LessThan5000              -> bool
        - Count of Question marks   -> int
        - 1Question                 -> bool
        - Count of Exclamation marks-> int
        """
    charcount = '{}_CharCount'.format(col_name)
    df[charcount] = df[col_name].str.len()
    lessthan = '{}_LessThan5000'.format(col_name)
    df[lessthan] = df[charcount] < 5000
    wordcount = '{}_WordCount'.format(col_name)
    df[wordcount] = df[col_name].str.split().str.len()
    wordcount_bucket = '{}_WordCount_Bucket'.format(col_name)
    bucket_conditions = [
        (df[wordcount]>0) & (df[wordcount]<=5),
        (df[wordcount]>5) & (df[wordcount]<=12),
        (df[wordcount]>12),
    ]
    bucket_options = ['Simple', 'Normal', 'Complex']
    df[wordcount_bucket] = np.select(bucket_conditions, bucket_options, default='Empty')
    questionmark = '{}_Questionmarks'.format(col_name)
    df[questionmark] = df[col_name].str.count('\?')
    onequestion = '{}_1Question'.format(col_name)
    df[onequestion] = df[questionmark] == 1
    exclamationmark = '{}_Exclamationmarks'.format(col_name)
    df[exclamationmark] = df[col_name].str.count('!')

def df_get_tokens(df, col_name, n_gram=1):
    """ Create tokens of 'col_name' returns tokenized_text """
    def _token_creator(sentence):
        replaced_punctation = list(map(lambda token: re.sub(r'[^\wa-zA-Z0-9!?]+', '', token), sentence))
        removed_punctation = list(filter(lambda token: not token.isdigit(), replaced_punctation))
        removed_empty = list(filter(None, removed_punctation))
        
        replace_ = list(map(lambda token: re.sub(r'^_|(\d)+(_$|)|_\W|\W_|_$', '', token), removed_empty))
        replace_ = list(map(lambda token: re.sub(r'^_|_$', '', token), replace_))
        removed_empty = list(filter(None, replace_))
        
        return removed_empty
    
    if n_gram == 1:
        df['tokenized_text'] = list(map(nltk.word_tokenize, df[col_name]))
    else:
        df['tokenized_text'] = df[col_name].apply(lambda x: ['_'.join(ng) for ng in nltk.everygrams(nltk.word_tokenize(x), 1, n_gram)])
    df['tokenized_text'] = list(map(_token_creator, df.tokenized_text))

def df_stem_words(df, col_name):
    # lemm = nltk.stem.WordNetLemmatizer()
    # df['lemmatized_text'] = list(map(lambda sentence: list(map(lemm.lemmatize, sentence)), df[col_name]))

    # p_stemmer = nltk.stem.porter.PorterStemmer()
    d_stemmer = nltk.stem.snowball.DanishStemmer()
    df['stemmed_text'] = list(map(lambda sentence: list(map(d_stemmer.stem, sentence)), df[col_name]))

def get_bigrams(tokens, chi_sq=500):
    from nltk.collocations import BigramCollocationFinder
    from nltk.metrics import BigramAssocMeasures
    try:
        bigram_finder = BigramCollocationFinder.from_words(tokens)
        bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, chi_sq)

        bi_list = []
        for bigram_tuple in bigrams:
            x = "%s %s" % bigram_tuple
            bi_list.append(x)
    except:
        print('Error happend')
        bi_list = ['error happend']
    finally:
        result = [' '.join([w for w in x.split()]) for x in bi_list]# if x.lower() not in stopwords.words('english') and len(x) > 8]
    return result

def df_bigrams(df, col_name):
    df['bigrams'] = list(map(get_bigrams, df.tokenized_text))

def get_trigrams(tokens, chi_sq=500):
    try:
        trigram_finder = TrigramCollocationFinder.from_words(tokens)
        trigrams = trigram_finder.nbest(TrigramAssocMeasures.chi_sq, chi_sq)

        tri_list = []
        for trigram_tuple in trigrams:
            x = "%s %s %s" % trigram_tuple
            tri_list.append(x)
    except:
        # print('Error happend')
        tri_list = ['error happend']

    finally:
        result = [' '.join([w for w in x.split()]) for x in tri_list]# if x.lower() not in stopwords.words('english') and len(x) > 8]
    return result

def df_trigrams(df, col_name):
    df['trigrams'] = list(map(get_trigrams, df.tokenized_text))

def getStopWords():
    """ Returns a list with all stopwords """
    import os
    cur_dir = os.getcwd()

    dk_addition = [line.rstrip('\n') for line in open(os.path.join(cur_dir,'utils','danish_stopwords.txt'), encoding="utf-8")]                            # danish stopword list

    customer_specific_words = [line.rstrip('\n') for line in open(os.path.join(cur_dir,'utils','stopwords_lda_customer_specific.txt'), encoding="utf-8")]  # customer specific
    dk_addition.extend(customer_specific_words)

    stopwords_1gram = [line.rstrip('\n') for line in open(os.path.join(cur_dir,'utils','stopwords_1gram.txt'), encoding="utf-8")]                         # stopwords 1grams
    dk_addition.extend(stopwords_1gram)

    stopwords_2gram = [line.rstrip('\n') for line in open(os.path.join(cur_dir,'utils','stopwords_2gram.txt'), encoding="utf-8")]                         # stopwords 2grams
    dk_addition.extend(stopwords_2gram)
    
    stopwords_3gram = [line.rstrip('\n') for line in open(os.path.join(cur_dir,'utils','stopwords_3gram.txt'), encoding="utf-8")]                         # stopwords 3grams
    dk_addition.extend(stopwords_3gram)
    
    # nltk
    stopwords = nltk.corpus.stopwords.words('danish')
    stopwords.extend(dk_addition)
    stopwords = list(set(stopwords))
    return stopwords

# READ HERE
# https://stackoverflow.com/questions/21844546/forming-bigrams-of-words-in-list-of-sentences-with-python/21844680#21844680
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.tokenize import WordPunctTokenizer
# from nltk.collocations import BigramCollocationFinder
# from nltk.metrics import BigramAssocMeasures

# def get_bigrams(myString):
#     tokenizer = WordPunctTokenizer()
#     tokens = tokenizer.tokenize(myString)
#     stemmer = PorterStemmer()
#     bigram_finder = BigramCollocationFinder.from_words(tokens)
#     bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)

#     for bigram_tuple in bigrams:
#         x = "%s %s" % bigram_tuple
#         tokens.append(x)

#     result = [' '.join([stemmer.stem(w).lower() for w in x.split()]) for x in tokens if x.lower() not in stopwords.words('english') and len(x) > 8]
#     return result


def df_remove_stopwords(df, col_name):
    """ Removes stopwords based on a known set of stopwords
    available in the nltk package. In addition, we include our made up word in here.
    Input is a dataframe with a column, containing list of lists.
    """
    stopwords = getStopWords()

    # Create column 'stopwords_removed' on df
    df['stopwords_removed'] = list(map(lambda doc: [word for word in doc if word not in stopwords], df[col_name]))

def document_to_bow(df, col_name, dictionary):
    "Make a BoW for every text"
    df['bow'] = list(map(lambda doc: dictionary.doc2bow(doc), df[col_name]))

def PrepareDictionary(df, col_name, no_above=1, no_below=0, log=None):
    """ Return Dictionary and Corpus to be used for training
    """
    from gensim.corpora import Dictionary
    # Create dictionary with words from df_scope (the total dataset)
    dictionary = Dictionary(documents=df[col_name].values)
    words_preprocess = len(dictionary.values())
    
    dictionary.filter_extremes(no_above=no_above, no_below=no_below, keep_tokens=['?'])

    dictionary.compactify()  # Reindexes the remaining words after filtering
    words_postprocess = len(dictionary.values())
    
    document_to_bow(df, col_name, dictionary)
    corpus = df.bow
    
    if log:
        log['words_preprocess'] = words_preprocess
        log['words_postprocess'] = words_postprocess

    return dictionary, corpus

def df_lda_features(lda_model, df, bow=None):
    import numpy
    def document_to_lda_features(lda_model, document):
        """ Transforms a bag of words document to features.
        It returns the proportion of how much each topic was
        present in the document.
        """
        topic_importances = lda_model.get_document_topics(document, minimum_probability=0)
        topic_importances = numpy.array(topic_importances)
        return topic_importances[:,1]

    if not bow:
        bow = df.bow

    df['lda_features'] = list(map(lambda doc: document_to_lda_features(lda_model, doc), bow))

def get_lda_topics(df, model, num_topics, topn=10):
    import numpy
    import pandas as pd

    mean_df_lda_features = df['lda_features'].mean()

    word_dict = {}
    for i in sorted(numpy.argsort(mean_df_lda_features)[-num_topics:]):
        words = model.show_topic(i, topn = topn)
        word_dict['Topic # ' + '{:02d}'.format(i)] = [i[0] for i in words]
    return pd.DataFrame(word_dict)

def get_topics_and_probability(df, lda_model, n_topics, n_topwords):
    import numpy
    import pandas as pd

    mean_df_lda_features = df['lda_features'].mean()

    def get_topic_and_prob(lda_model, topic_id, nr_top_words=10):
        id_tuples = lda_model.print_topic(topic_id, topn=nr_top_words)
        id_list = []
        for i in id_tuples.split("+"):
            i0, i1 = i.split('*')
            id_list.append((float(i0.strip()),i1.split('"')[1]))
        return id_list

    topics_dist = pd.DataFrame()

    for x in sorted(numpy.argsort(mean_df_lda_features)[-n_topics:]):
        top_words = get_topic_and_prob(lda_model, x, n_topwords) #get_topic_top_words(LDAmodel, x)
        # print("Topic: {0}\tProb: {1:.3f}, Words: {2}.".format(x, RequestTopicDistribution.item(x), top_words)) #(x, ", ".join(top_words)))
        row = {
            'Topic': x,
            'Topic_Distribution': mean_df_lda_features.item(x),
            'TopWords': top_words,
        }
        topics_dist = topics_dist.append(row, ignore_index=True)
    
    topics_dist.index.names = ['topic_index']
    topics_dist['topic_id'] = topics_dist.index

    df_topwords = topics_dist.apply(lambda x: pd.Series(x['TopWords']),axis=1).stack()#.reset_index(level=1, drop=True)
    df_topwords.name = 'prob_words'
    df_topwords = pd.DataFrame(df_topwords)
    df_topwords[['Probability_Word','Word']] = df_topwords['prob_words'].apply(pd.Series)
    df_topwords.index.names = ['topic_index','word_index']
    df_topwords['topic_id'] = df_topwords.index.get_level_values('topic_index')
    df_topwords['Word_Rank'] = df_topwords.index.get_level_values('word_index')
    
    # return df_topwords, topics_dist

    df = pd.merge(df_topwords, topics_dist, on='topic_id')
    df.drop(['TopWords', 'topic_id','prob_words'], axis=1, inplace=True)

    return df

# def lda_predict_df(df, col_name, lda_model, dictionary, lda_topic_name_dict=None):
#     """ Make it possible to clean the df if that hasnt happend yet
#         It should be possible to select the dataframe to be predicted
        
#         The current function assumes that data is clean and has a column named stemmed_text"""
# #     for index, score in sorted(LDAmodel_lang[bow_vector], key=lambda tup: -1*tup[1]):
# #         print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
    
#     df['bow'] = list(map(lambda doc: dictionary.doc2bow(doc), df[col_name]))
#     if lda_topic_name_dict is None:
#         df['prediction'] = df['bow'].apply(PredictTopicFromBOW,lda_model=lda_model)
#         df[['pred_probability','pred_index']] = pd.DataFrame(df.prediction.values.tolist(), index= df.index)
#     else:
#         df['prediction'] = df['bow'].apply(PredictTopicFromBOW,lda_model=lda_model, lda_topic_name_dict=lda_topic_name_dict)
#         df[['pred_probability','pred_index','pred_label']] = pd.DataFrame(df.prediction.values.tolist(), index= df.index)
#     df.drop(['prediction'], axis=1)

   
def lda_predict_string(text, LDAmodel, dictionary, lda_topic_name_dict=None):
    
    bow_vector = dictionary.doc2bow(lda_preprocess_string(text))
    index, score = sorted(LDAmodel[bow_vector], key=lambda tup: -1*tup[1])[0]
    if lda_topic_name_dict is None:
        # lda_topic_name_dict = [str(i) for i in range(dictionary.values())]
        return score, index, LDAmodel.print_topic(index, 5)
    else:
        return score, lda_topic_name_dict[index], LDAmodel.print_topic(index, 5)

def lda_predict_string_dict(text, LDAmodel, dictionary, lda_topic_name_dict=None):
    bow_vector = dictionary.doc2bow(lda_preprocess_string(text))
    best_prediction = 0.0001
    prediction_index = None
    for index, score in sorted(LDAmodel[bow_vector], key=lambda tup: -1*tup[1]):
        if score > best_prediction:
            best_prediction = score
            prediction_index = index
    if lda_topic_name_dict is None:
        return best_prediction, prediction_index, LDAmodel.print_topic(prediction_index, 5), prediction_index, index
    else:
        return best_prediction, lda_topic_name_dict[prediction_index], LDAmodel.print_topic(prediction_index, 5), prediction_index, index

# def PredictTopicFromBOW(bow_vector, lda_model, lda_topic_name_dict=None):
#     "Input a string prepared bow-vector"
#     best_prediction = 0
#     prediction_index = None
#     for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
#         if score > best_prediction:
#             prediction_index = index
#             best_prediction = score

#     if lda_topic_name_dict is None:
#         pred = [best_prediction, prediction_index]#, lda_model.print_topic(index, 5)
#     else:
#         pred = [best_prediction, prediction_index, lda_topic_name_dict[prediction_index]]#, lda_model.print_topic(index, 5)
#     return pred

def df_lda_preprocessing(df, col_name, remove_stopwords=True, add_features=False):
    """ All the preprocessing steps for LDA are combined in this function.
    All mutations are done on the dataframe itself. So this function returns
    nothing.

    This method handles _preprocessing_
    Steps to take after this
    1. Create dictionary
    2. Create BoW
    3. Train model
    """
    df['text'] = df[col_name] # Create a copy of the input col_name: text
    
    # df_clean_sting(df, 'text') # Clean the text from col_name # TEST FJERN RENGØRING

    # Test other way of handling strings
    df_simple_clean_string(df, 'text')

    if add_features:
        df_make_features_from_string(df, 'text') # Add features

    # This is a hack soly for the scope of this project to concat ThreadSubject
    # When the message is initiated by the Member
    if col_name == 'SignalMessageBodyClean':
        df_aka = df.copy(deep=True)
        # df_aka['text_1'] = df_aka['ThreadSubject']
        # df_clean_sting(df_aka, 'ThreadTopic')
        df_simple_clean_string(df_aka, 'ThreadTopic')

        df['text'] = (df['text'] +' '+df_aka['ThreadTopic']).where(df['IsFirstMessageInthread']==1,df['text'])

    df_get_tokens(df, 'text') # Returns col: tokenized_text

    # df_stem_words(df, 'tokenized_text') # Returns col: stemmed_text

    df_bigrams(df, 'tokenized_text') # Returns bigrams
    df_trigrams(df, 'tokenized_text') # Returns trigrams

    df['ngrams'] =  df['tokenized_text'] + df['bigrams'] + df['trigrams']

    if remove_stopwords:
        df_remove_stopwords(df, 'ngrams') # returns stopwords_removed




def lda_preprocess_string(text):
    "Takes a string, and prepares in a format to be taken by the LDAmodel for prediction"

    def remove_standard_greetings(string):
        """ Removes
            Dear Name
            Dear Name Last
            Dear Name Middle Last
            and the newline right after
        """
        # Remove all start greeetings + name
        reg_rm_dear = r'^^((\n*\s*|\n*\S*\s*)(\b|\b\w)(kære|hej|hejsa|hello|hi|dear|til)\s?(\w[a-zæøåA-ZÆØÅ\w-]{2,}\s\w[a-zæøåA-ZÆØÅ\w.-]{0,}\s\w[a-zæøåA-ZÆØÅ\w-]{2,}|\w[a-zæøåA-ZÆØÅ\w-]{2,}\s\w[a-zæøåA-ZÆØÅ\w-]{0,}|\w[a-zæøåA-ZÆØÅ\w.-]{2,})?(|\n))'
        string = re.sub(reg_rm_dear,'', str(string.lower()))
        
        # Remove all after curtisy + name
        reg_rm_regads = r'(\b(med\svenlig\shilsen|de\sbedste\shilsner|venlig\shilsen|mvh|hilsen|vh|best\sregards|with\skind\sregards|kind\sregards|with\sregards|regards|sent\sfrom\smy\s).*)'
        start_of_regards = re.findall(reg_rm_regads, string)
        
        if len(start_of_regards) > 0 and start_of_regards[0] is not None:
            start_of_regards = list(start_of_regards[0])[0]
            string = string[:string.find(start_of_regards)].strip()
        
        return string

    def clean_text(text):
        "Parse HTML using BeautifulSoup and return the text"
        text = re.sub('</p><p>','</p>.\n<p>', str(text))
        text = BeautifulSoup(text, 'html.parser').get_text()
        # text = re.sub(r'^\n','',str(text))
        text = text.strip()
        text = remove_standard_greetings(text)
        text = re.sub(r'^\n','',str(text))
        text = re.sub(r'\xa0','',str(text))
        text = text.strip()
        words = text.lower().split()
        return ' '.join(words)
    
    def replace_specific_words(string):
        dictionary = {
        'vedr.': 'vedrørende',
        'vedr ': 'vedrørende ',
        'mdr.s': 'måneders',
        'mdr': 'måneder',
        'medl.': 'medlem',
        'o.s.v': 'og så videre',
        'm.m.': 'med mere.',
        'att ': 'til ',
        'att: ': 'til ',
        'att.: ': 'til ',
        'i.f.m': 'i forbindelse med',
        'ph.d.': 'philosophiaedoctor',
        ' d.': ' den',
        'ang.': 'angående',
        ' re ': ' svar ',
        ' re:': ' svar',
        'bh.': 'bedste hilsener',
        }
        
        for word, initial in dictionary.items():
            string = string.replace(word.lower(), initial)
        
        # Remove e-mail, Specify the number of replacements by changing the 4th argument
        string = re.sub(r'[\w\.-]+@[\w\.-]+','',string,0)
        
        # Remove words that are smaller than 2 letters
        string = re.sub(r'\b\w{1,2}\b', '',string)
        
        return string

    text = clean_text(text) # "Parse HTML using BeautifulSoup and return the text"
    text = replace_specific_words(text) # replace accronyms
    
    stopwords = nltk.corpus.stopwords.words('danish')
    dk_addition = [line.rstrip('\n') for line in open('data/danish_stopwords.txt')]
    dk_start_stop_message_words = ['kære','hilsen', 'mvh', 'venlig','tusind','tak']
    dk_addition.extend(dk_start_stop_message_words)
    
    customer_specific_words = ['akasse','aka','akademikernes','!','akadk']
    dk_addition.extend(customer_specific_words)
        
    stopwords.extend(dk_addition)
    stopwords = list(set(stopwords))
    
    
    # lda_get_good_tokens
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    # get_good_tokens
    text = list(map(lambda t: re.sub(r'[^\wa-zA-Z0-9!?]+', '', t), text))
    text = list(filter(lambda token: not token.isdigit(), text))

    # remove_stopwords(df)
    text = list(filter(lambda t: t not in stopwords, text))
    text = list(filter(None, text))    
    
    # stem_words(df)
    lemm = nltk.stem.WordNetLemmatizer()
    text = list(map(lambda t: lemm.lemmatize(t), text))

    p_stemmer = nltk.stem.porter.PorterStemmer()
    text = list(map(lambda t: p_stemmer.stem(t), text))
    
    return text
