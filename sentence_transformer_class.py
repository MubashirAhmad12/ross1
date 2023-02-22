from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd


class sentence_transformer:
    def __init__(self, filename, test_sentence):
        self.filename = filename
        self.test_sentence = test_sentence
    
    def similarity(self):
        df = pd.read_csv(self.filename)
        # Convert the content column of the dataframe to list
        cv_content_sentences_list=df['Resume_str'].values.tolist()
        # model = SentenceTransformer('stsb-roberta-large')

        model = torch.load("sentence_transformer.pkl")
        #Compute embedding for both lists
        embedding_1= model.encode(self.test_sentence, convert_to_tensor=True)
        embedding_2 = model.encode(cv_content_sentences_list, convert_to_tensor=True)

        similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
        df['similarities']=similarity[0]
        df = df.sort_values("similarities", ascending=False).head(3)
        return df


def main():
    filename = 'data/Resume_small1.csv'
    Description = str(input('Enter the Description: '))
    
    p1 = sentence_transformer(filename, Description)
    sorted_df = p1.similarity()
    print(sorted_df)
    
if __name__ == "__main__":
    main()
