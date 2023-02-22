from PyPDF2 import PdfReader
import csv
import openai
import pandas as pd
import numpy as np
from getpass import getpass
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

openai.api_key = getpass()

def get_cv_embedd(filename):    
    # Read the csv file that contain the csv filenames and content in it
    df = pd.read_csv(filename)
    # Get the embedding of the content of each each and save in a seperate column having name of embedding
    df['embedding'] = df['Resume_str'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    #  Save the embeddings in the csv format so that we don't need to recompute the embeddings again
    df.to_csv('resume_embeddings.csv')
    return df


def similarity_check(embedding_file):
    df = pd.read_csv(embedding_file)
    # Change the type of embedding column to numpy array
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    # Enter a description by the recruiter
    # search_term = input('Enter a description by the recruiter: ')
    search_term = "I am looking for HR who is expert in Desktop Publishing, Web page development, WordPerfect, and database Management"
    
    #  Find the embedding of the searched description
    search_term_vector = get_embedding(search_term, engine="text-embedding-ada-002")
    # Find the similarity of the search description embedding with the CV's embedding
    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
    #  Extract the top 3 CV's
    df = df.sort_values("similarities", ascending=False).head(3)
    print(df)
    return df


def text_extractor_from_cv(filepath):
    reader = PdfReader(filepath)
    number_of_pages = len(reader.pages)
    text_page = ""
    for i in range(0, number_of_pages):
        page = reader.pages[i]
        text = page.extract_text()
        text_page = text_page +" " + text
    text_page = text_page.replace("\n"," ")
    # count = 0
    # with open('extract.csv', 'w') as f:
    #     f.write("filename"+","+"content")
    #     f.write(text_page)

    with open('extract.txt', 'w') as f:
        f.write(text_page)
    return text
    

def main():
    csv_file = 'C:/Users/XEVEN-DEV/Documents/xeven/ross_project/dataset/resume_dataset/Resume/Resume_small1.csv'
    embedding_file = 'resume_embeddings.csv'
    df_embedd = get_cv_embedd(csv_file)
    df_similarity = similarity_check(embedding_file)
    # filepath="C:/Users/XEVEN-DEV/Documents/xeven/ross_project/dataset/resume_dataset/data/data/ACCOUNTANT/10554236.pdf"
    # text = text_extractor_from_cv(filepath)

    
if __name__ == "__main__":
    main()

