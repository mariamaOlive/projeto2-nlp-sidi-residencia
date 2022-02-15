from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def join_docs(df_text, column1, column2, unique = False):

    data = []
    for i in range(0, len(df_text)):
        data.append(' '.join(df_text[column1][i]))
        data.append(' '.join(df_text[column2][i]))

    if unique == True:
        data = list(set(data))

    return data


###Bag of Words###
def get_bow(doc1, doc2):
    
    vectorizer = CountVectorizer()
    
    text_list1 = ' '.join(doc1)
    text_list2 = ' '.join(doc2)
    
    text_list = [text_list1, text_list2]

    vector = vectorizer.fit_transform(text_list)
    
    cosine_similarities = cosine_similarity(vector[0], vector[1])
    
    return cosine_similarities[0][0]


def apply_bow(df, len_pipeline):

    df_bow = pd.DataFrame()
    for index in range(len_pipeline):
    
        df_bow[f'bow{index}'] = df.apply(lambda row: get_bow(row[f'doc1_pipeline{index}'], row[f'doc2_pipeline{index}']), axis=1)

    return df_bow



###TF-IDF###

def calculate_tf_idf(df, column1, column2):
    
    data = join_docs(df, column1, column2)
      
    tf_idf = TfidfVectorizer().fit_transform(data)
    
    return tf_idf

def get_tf_idf(tf_idf, index):
    
    index1 = 2*index
    index2 = 2*index + 1
    
    cosine_similarities = cosine_similarity(tf_idf[index1], tf_idf[index2])
    
    return cosine_similarities[0][0]

def apply_tf_idf(df, len_pipeline):
    
    df_tf_idf = pd.DataFrame()

    for index in range(len_pipeline):
    
        tf_idf = calculate_tf_idf(df, f'doc1_pipeline{index}', f'doc2_pipeline{index}')

        tf_idf_list = []
        for index_df in range(len(df)):
            tf_idf_list.append(get_tf_idf(tf_idf, index_df))
        
        df_tf_idf[f'tf_idf{index}'] = tf_idf_list
    
    return df_tf_idf
        

''' 
    for param, index in zip(pre_processing_list, range(len(pre_processing_list))):
        
        df[f"doc1_pipeline{index}"] = df["doc1"].apply(lambda x: pre_process(x, **param))
        df[f"doc2_pipeline{index}"] = df["doc2"].apply(lambda x: pre_process(x, **param))
'''


