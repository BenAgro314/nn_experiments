#!/usr/bin/env python3
import string
import nltk
from nltk.tokenize import WhitespaceTokenizer
import pickle
import matplotlib.pyplot as plt
from wordfreq import word_frequency
from sklearn.feature_extraction.text import CountVectorizer


def get_text():
    try:
        stream = open("data/transcripts.pkl", "rb")
    except FileNotFoundError:
        print("ERROR: data is not located at data/transcipts.pkl")

    transcripts =pickle.load(stream)

    text = []


    freq_lex_dict = {}
    total = 0
    table = str.maketrans("", "", string.punctuation)
    for transcript in transcripts.values():
        text.append(transcript)
        """
        transcript = transcript.lower()
        transcript = transcript.split("-")
        transcript = " ".join(transcript)
        words = transcript.split()
        words = [w.translate(table) for w in words]
        for word in words:
            total +=1
            if word not in freq_lex_dict:
                freq_lex_dict[word] = 1
            else:
                freq_lex_dict[word] += 1
        """

    text = " ".join(text)

    tokenizer_w = WhitespaceTokenizer()
    tokenized_list = tokenizer_w.tokenize(" ".join(text)) 
    return tokenized_list

    """
    stop = set(nltk.corpus.stopwords.words('english'))
    exclude = set(string.punctuation) 
    lemma = nltk.stem.wordnet.WordNetLemmatizer()
    def clean(doc):
        stop_free = " ".join([i for i in doc.split() if i not in stop])
        punc_free = "".join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized
    """

    #text = clean(text).lower().split()
    #return text


"""
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text)
words = vectorizer.get_feature_names()
print(X.toarray())
freq = {}
for word, count in zip(words, X.toarray()[0]):
    freq[word] = count

sorted_freq = sorted(freq.items(), key = lambda x: x[1], reverse = True)
print(sorted_freq[:100])
"""

"""
lex_freq = sorted(freq_lex_dict.items(), key=lambda x: x[1], reverse = True)
lex_freq = [(word, num/total) for word, num in lex_freq]

freq_diffs = []
for item in lex_freq:
    word = item[0]
    actual_freq = word_frequency(word, "en")
    if actual_freq == 0:
        continue
    freq_diffs.append((word, item[1]/actual_freq))

freq_diffs = sorted(freq_diffs, key = lambda x: x[1], reverse = True)

fig, ax = plt.subplots(2, 1)
num = 25
ax[0].bar(
    [item[0] for item in freq_diffs[:num]],
    [item[1] for item in freq_diffs[:num]]
)
ax[1].bar(
    [item[0] for item in lex_freq[:num]],
    [item[1] for item in lex_freq[:num]]
)
fig.set_size_inches(30, 10.5)
plt.savefig("freq_diff.png", dpi = 100)
"""