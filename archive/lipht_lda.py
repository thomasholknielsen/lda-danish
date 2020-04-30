import pandas as pd
import nltk

import re

from utils.lipht_data import df_clean_sting
from utils.lipht_lda_utils import df_make_features_from_string, df_stem_words, df_get_tokens, df_remove_stopwords, document_to_bow, df_bigrams, df_trigrams


def df_lda_preprocessing(df, col_name, remove_stopwords=True):
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
    df_clean_sting(df, 'text') # Clean the text from col_name
    df_make_features_from_string(df, 'text') # Add features

    df_get_tokens(df, 'text') # Returns col: tokenized_text

    df_stem_words(df, 'tokenized_text') # Returns col: stemmed_text

    df_bigrams(df, 'stemmed_text') # Returns bigrams
    df_trigrams(df, 'stemmed_text') # Returns trigrams

    df['ngrams'] = df['bigrams'] + df['trigrams'] + df['stemmed_text']

    if remove_stopwords:
        df_remove_stopwords(df, 'ngrams')




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
        reg_rm_dear = r'^(\n*\s*|\n*\S*\s*)(\b|\b\w)(kære|hej|hejsa|hello|hi|dear)\s?(\w[a-zA-Z\w.-]{2,}|\w[a-zA-Z\w.-]{2,}\s\w[a-zA-Z\w.-]{0,}|\w[a-zA-Z\w.-]{2,}\s\w[a-zA-Z\w.-]{0,}\s\w[a-zA-Z\w.-]{2,})?.?[\n]?'
        string = re.sub(reg_rm_dear,'', str(string.lower()))
        
        # Remove all after curtisy + name
        reg_rm_regads = r'(\b(med\svenlig\shilsen|de\sbedste\shilsner|venlig\shilsen|mvh|hilsen|venlig|best\sregards|with\skind\sregards|kind\sregards|with\sregards|regards|sent\sfrom\smy\s).*)'
        start_of_regards = re.findall(reg_rm_regads, string)
        
        if len(start_of_regards) > 0 and start_of_regards[0] is not None:
            start_of_regards = list(start_of_regards[0])[0]
            string = string[:string.find(start_of_regards)].strip()
        
        return string

    def clean_text(text):
        from bs4 import BeautifulSoup
        "Parse HTML using BeautifulSoup and return the text"
        text = re.sub('</p><p>','</p>.\n<p>', str(text))
        text = BeautifulSoup(text, 'html.parser').get_text()
        text = re.sub(r'^\n','',str(text))
        text = remove_standard_greetings(text)
        text = re.sub(r'^\n','',str(text))
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


