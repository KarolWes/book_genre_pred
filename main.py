import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# additional downloads
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def nlp_cleaning(data, output_filename):
    english_stops = set(stopwords.words("english"))
    wl = WordNetLemmatizer()
    new_col_stems = []
    for summary in data["summary"]:
        print(".", end="")
        tokens = word_tokenize(summary.lower())  # split into tokens
        letters_only = [token for token in tokens if token.isalpha()]  # remove non-alpha
        no_stops = [token for token in letters_only if token not in english_stops]  # remove stop words
        stems = [wl.lemmatize(token) for token in no_stops]  # remove plural forms
        stems = [wl.lemmatize(token, 'v') for token in stems]  # change verbs to base form
        new_col_stems.append(stems)
    data["clean"] = new_col_stems
    data.to_csv(output_filename)


def string_to_list(s):
    s = s[1:len(s) - 1]
    tmp = s.split()
    ans = []
    for label in tmp:
        ans.append(label[1:len(label) - 2])
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


def graph(data):
    plt.figure(figsize=(10, 10))
    g = sns.countplot(x="genre", data=data)
    plt.xticks(rotation=45)
    fig = g.get_figure()
    fig.savefig("wykres1.png")
    plt.show()


def words_dict_by_genre(data):
    dict = sum_counter(data)
    for key in dict.keys():
        print(key + ": " + str(dict[key].most_common(10)))


def fit_best(X_train, X_test, y_train, y_test, c):
    model = OneVsRestClassifier(LogisticRegression(C=c))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    ac = accuracy_score(pred, y_test)
    f1 = f1_score(pred, y_test, average="macro")
    ans = [c, ac, f1]
    return ans


def fit(X_train, X_test, y_train, y_test, old_labels, classifier):
    ans = []
    models = [KNeighborsClassifier(), LogisticRegression(), MultinomialNB(), SVC()]
    graph, axes = plt.subplots(len(models) // 2, 2, figsize=(18, 12))
    graph.suptitle(str(classifier).split('.')[-1][:-2], size=24)
    graph.tight_layout(pad=5.0)
    for i, model in enumerate(models):
        model = classifier(model)
        start = time.process_time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fin = time.process_time()
        ac = accuracy_score(pred, y_test)
        f1 = f1_score(pred, y_test, average="macro")

        ans.append([str(model), ac, f1, (fin - start) * 1000])

        cfm = confusion_matrix(y_test, pred)

        plt.figure(figsize=(10, 10))
        fig = sns.heatmap(cfm / cfm.sum(axis=1)[:, None] * 100, annot=True, cmap='Greens', vmax=100)
        fig.set_title(str(model) + "[%]")
        fig.set_xlabel("Predicted")
        fig.set_ylabel("Real")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        fig.xaxis.set_ticklabels(old_labels)
        fig.yaxis.set_ticklabels(old_labels)
        fig = fig.get_figure()
        fig.savefig(str(model) + ".png")
        fig = sns.heatmap(cfm / cfm.sum(axis=1)[:, None] * 100, ax=axes[i // 2, i % 2], annot=True, cmap='Greens',
                          vmax=100)
        fig.tick_params(labelrotation=45)
        fig.xaxis.set_ticklabels(old_labels)
        fig.yaxis.set_ticklabels(old_labels)
        title = str(model).split('=')[1][:-3]
        fig.title.set_text(title)

    df = pd.DataFrame(ans, columns=["model", "accuracy", "F1 score", "time ms"])
    graph.savefig("grid_ovo.png")
    return df


if __name__ == '__main__':
    # data = pd.read_csv("data.csv", index_col=0)
    # print(data.head())
    # print("shape of the data frame ", data.shape)
    # print("does it contains missing values?\n", data.isna().any(), "\n_______")
    # graph(data)
    # nlp_cleaning(data, "out.csv")
    data = pd.read_csv("out.csv")
    # words_dict_by_genre(data)
    # print(data.head())
    encoder = LabelEncoder()
    data["genre"] = encoder.fit_transform(data["genre"])
    old_labels = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    # X_data = TfidfVectorizer(max_df=0.95).fit_transform(data["clean"])
    X_data = CountVectorizer().fit_transform(data["clean"])
    X_resampled, y_resampled = RandomOverSampler(random_state=119).fit_resample(X_data, data["genre"])
    # X_train, X_test, y_train, y_test = train_test_split(X_data, data["genre"], test_size=0.25, random_state=79)
    # dist = Counter(y_test)
    # genre_distribution = np.array([dist[i] for i in range(len(dist))])
    # print(dist)
    # res = []
    # start = time.process_time()
    # for r in range(3):
    #     X_train, X_test, y_train, y_test = train_test_split(X_data, data["genre"], test_size=0.25, random_state=r*17+13)
    #     for c in range(1,11):
    #         print(r, " ", c)
    #         res.append(fit_best(X_train, X_test, y_train, y_test, c/10))
    # df = pd.DataFrame(res, columns=["reg. param.", "accuracy", "F1 score"])
    # # df = df.groupby('reg. param.').mean()
    # fin = time.process_time()
    # print((fin - start))
    X_train, X_test, y_train, y_test = train_test_split(X_data, data["genre"], test_size=0.25, random_state=79)
    print("___")
    print(fit(X_train, X_test, y_train, y_test,old_labels, OneVsOneClassifier))
    print("___")

    #df = fit(X_train, X_test, y_train, y_test, old_labels)
    # print(df)
