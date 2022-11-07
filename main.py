import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# additional downloads
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def nlp_cleaning(data):
    english_stops = set(stopwords.words("english"))
    wl = WordNetLemmatizer()
    new_col_stems = []
    new_col_dicts = []
    for summary in data["summary"]:
        print(".", end="")
        tokens = word_tokenize(summary.lower())  # split into tokens
        letters_only = [token for token in tokens if token.isalpha()]  # remove non-alpha
        no_stops = [token for token in letters_only if token not in english_stops]  # remove stop words
        stems = [wl.lemmatize(token) for token in no_stops]  # remove plural forms
        stems = [wl.lemmatize(token, 'v') for token in stems]  # change verbs to base form
        word_dict = Counter(stems)
        new_col_dicts.append(word_dict)
        new_col_stems.append(stems)
    data["clean"] = new_col_stems
    data["word_dictionary"] = new_col_dicts


def string_to_list(s):
    s = s[1:len(s)-1]
    tmp = s.split()
    ans = []
    for label in tmp:
        ans.append(label[1:len(label)-2])
    return ans


def sum_counter(data):
    genre = set(data["genre"])
    genre_dict = {g: [] for g in genre}
    genre_dict_counter = {}
    for _, row in data.iterrows():
        genre_dict[row["genre"]].append(Counter(string_to_list(row["clean"])))
    for key in genre_dict.keys():
        genre_dict_counter[key] = sum(genre_dict[key], Counter())
    return genre_dict_counter


if __name__ == '__main__':
    data = pd.read_csv("out.csv", index_col=0)
    print(data.head())
    print("shape of the data frame ", data.shape)
    print("does it contains missing values?\n", data.isna().any(), "\n_______")
    # plt.figure(figsize=(10, 10))
    # g = sns.countplot(x="genre", data=data)
    # plt.xticks(rotation=45)
    # fig = g.get_figure()
    # fig.savefig("wykres1.png")
    # plt.show()
    # nlp_cleaning(data)
    # data.to_csv('out.csv')
    dict = sum_counter(data)
    for key in dict.keys():
        print(key + ": " + str(dict[key].most_common(10)))
    # print(data.head())
    # X_data = CountVectorizer.fit_transform(data["clean"])
    # X_train, X_test, y_train, y_test = train_test_split(X_data, data["genre"], test_size=0.25, random_state=79)