import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter


#additional downloads
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')


def nlp_cleaning(data):
    english_stops = set(stopwords.words("english"))
    wl = WordNetLemmatizer()
    new_col = []
    for summary in data["summary"]:
        print(".", end="")
        tokens = word_tokenize(summary.lower()) #split into tokens
        letters_only = [token for token in tokens if token.isalpha()] #remove non-alpha
        no_stops = [token for token in letters_only if token not in english_stops] #remove stop words
        stems = [wl.lemmatize(token) for token in no_stops] #remove plural forms
        stems = [wl.lemmatize(token, 'v') for token in stems] #change verbs to base form
        word_dict = Counter(stems)
        new_col.append(word_dict)
    data["dict"] = new_col


if __name__ == '__main__':
    data = pd.read_csv("data.csv", index_col=0)
    print(data.head())
    print("shape of the data frame ", data.shape)
    print("does it contains missing values?\n", data.isna().any(), "\n_______")
    plt.figure(figsize=(10, 10))
    g = sns.countplot(x="genre", data=data)
    plt.xticks(rotation=45)
    fig = g.get_figure()
    fig.savefig("wykres1.png")
    plt.show()
    # nlp_cleaning(data)
    # print(data.head())

