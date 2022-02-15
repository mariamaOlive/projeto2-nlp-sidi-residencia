from sklearn.feature_extraction.text import CountVectorizer
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

''' 
    for param, index in zip(pre_processing_list, range(len(pre_processing_list))):
        
        df[f"doc1_pipeline{index}"] = df["doc1"].apply(lambda x: pre_process(x, **param))
        df[f"doc2_pipeline{index}"] = df["doc2"].apply(lambda x: pre_process(x, **param))
'''