
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

filename = 'C:/Users/XEVEN-DEV/Documents/xeven/ross_project/dataset/resume_dataset/Resume/Resume_small1.csv'
test_sentence = "I am looking for HR who is expert in Desktop Publishing, Web page development, WordPerfect, and database Management"
df = pd.read_csv(filename)
# Convert the content column of the dataframe to list
cv_content_sentences_list=df['Resume_str'].values.tolist()
# model = SentenceTransformer('stsb-roberta-large')

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# model = torch.load("sentence_transformer_roberta_large.pkl")
#Compute embedding for both lists
embedding_1= model.encode(test_sentence, convert_to_tensor=True)
embedding_2 = model.encode(cv_content_sentences_list, convert_to_tensor=True)

similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
df['similarities']=similarity[0]
df = df.sort_values("similarities", ascending=False).head(3)
print(df)