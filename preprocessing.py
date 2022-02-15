#Imports
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer



#Global variables
other_punctuation = '—“”'  
stop_words = stopwords.words('english')
stop_words.append('’')
porter = PorterStemmer()
word_net_lemmatizer = WordNetLemmatizer()


#Function that removes punctuation 
def remove_punctuation(text):
    punctuation_free_doc = "".join([i for i in text if i not in string.punctuation+other_punctuation])
    return punctuation_free_doc


def remove_stopwords(list_words):
    filtered_words = [word for word in list_words if word not in stop_words]
    return filtered_words


def do_stemming(list_words):
    stem_text = [porter.stem(word) for word in list_words]
    return stem_text


def do_lemmatization(list_words):
    lemm_text = [word_net_lemmatizer.lemmatize(word) for word in list_words]
    return lemm_text


def pre_process(doc, basic_processing = False, no_stopwords = False, stemming = False, lema = False):

    final_doc = doc

    if basic_processing == True:
        
        final_doc = remove_punctuation(doc)
        final_doc = final_doc.lower()

    final_doc = nltk.word_tokenize(final_doc)

    if no_stopwords == True:
        final_doc = remove_stopwords(final_doc)    

    if stemming == True:
        final_doc = do_stemming(final_doc)

    if lema == True:
        final_doc = do_lemmatization(final_doc)

    return final_doc


#sem proces - pre_process(doc)
#com stopword - pre_process(doc, basic_processing = True)
#sem stopword - pre_process(doc, basic_processing = True, no_stopwords = True)
#com stemmizacao - pre_process(doc, basic_processing = True, stemming = True)
#com stemmizacao - pre_process(doc, basic_processing = True, no_stopwords = True, stemming = True)
#com lema - pre_process(doc, basic_processing = True, lema = True)
#com lema - pre_process(doc, basic_processing = True, no_stopwords = True, lema = True)