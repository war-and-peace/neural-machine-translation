import os
from collections import Counter, defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')


def png_file(title):
    return os.path.join('./images', title + '.png')


def print_stats(corpus, language='English'):
    total_count = sum([len(sentence.split()) for sentence in corpus])
    counter = Counter([word for sentence in corpus for word in sentence.split()])
    avg_sentence_length = sum([len(sentence.split()) for sentence in corpus]) / len(corpus)

    print(f'Stats for {language} corpus')
    print(f'\tTotal number of tokens: {total_count}')
    print(f'\tNumber of unique tokens: {len(counter)}')
    print(f'\tAverage number of words per sentence: {avg_sentence_length}')
    print(f'\tMost frequent tokens: {[x[0] for x in counter.most_common(10)]}')


def number_of_chars_per_sent_hist(corpus, source_name, language='English'):
    data = [len(sent) for sent in corpus]
    plt.figure(figsize=(10, 5))
    sns.histplot(data)
    plt.xlabel('chars per sentence')
    title = f'{source_name}. Distribution of characters per sentence in {language} corpus'
    plt.title(title)
    plt.savefig(png_file(title))
    plt.show()


def number_of_words_per_sent_hist(corpus, source_name, language='English'):
    data = [len(sent.split()) for sent in corpus]
    plt.figure(figsize=(10, 5))
    sns.histplot(data)
    plt.xlabel('words per sentence')
    title = f'{source_name}. Distribution of words per sentence in {language} corpus'
    plt.title(title)
    plt.savefig(png_file(title))
    plt.show()


def avg_word_len_per_sent_hist(corpus, source_name, language='English'):
    data = [np.mean([len(x) for x in sent.split()]) for sent in corpus]
    print('Max avg_word_len:', max(data), 'Min_avg_word_len', min(data))
    plt.figure(figsize=(15, 5))
    sns.histplot(data)
    plt.xlabel('average word length')
    title = f'{source_name}. Distribution of average word length in each sentence in {language} corpus'
    plt.title(title)
    plt.savefig(png_file(title))
    plt.show()


def get_max_avg_word_len_sent(corpus):
    return max([(sent, np.mean([len(x) for x in sent.split()])) for sent in corpus], key=lambda x: x[1])[0]


def most_common_stopwords_bar(corpus, source_name, language='English', top_n=20):
    stop_words = set(stopwords.words(language.lower()))
    dic = defaultdict(int)
    for sent in corpus:
        for word in sent.split():
            if word in stop_words:
                dic[word] += 1
    data = list(sorted(dic.items(), key=lambda x: -x[1]))
    x, y = [x[0] for x in data[:top_n]], [x[1] for x in data[:top_n]]
    plt.figure(figsize=(15, 5))
    sns.barplot(x, y)
    plt.xlabel('stopword')
    plt.ylabel('count')
    title = f'{source_name}. Most common stopwords in {language} corpus'
    plt.title(title)
    plt.savefig(png_file(title))
    plt.show()


def most_common_words_bar(corpus, source_name, language='English', top_n=20):
    stop_words = set(stopwords.words(language.lower()))
    dic = defaultdict(int)
    for sent in corpus:
        for word in sent.split():
            if word not in stop_words:
                dic[word] += 1
    data = list(sorted(dic.items(), key=lambda x: -x[1]))
    x, y = [x[0] for x in data[:top_n]], [x[1] for x in data[:top_n]]
    plt.figure(figsize=(15, 5))
    sns.barplot(x, y)
    plt.xlabel('word (not a stopword)')
    plt.ylabel('count')
    title = f'{source_name}. Most common words in {language} corpus'
    plt.title(title)
    plt.savefig(png_file(title))
    plt.show()


def most_common_ngrams_bar(corpus, source_name, language='English', n=2, top_n=10):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    data = [(word, sum_words[0, idx])
            for word, idx in vec.vocabulary_.items()]
    data = sorted(data, key=lambda x: -x[1])
    x, y = [x[0] for x in data[:top_n]], [x[1] for x in data[:top_n]]
    plt.figure(figsize=(15, 5))
    sns.barplot(x, y)
    plt.xlabel('ngram')
    plt.ylabel('count')
    title = f'{source_name}. Most common ngrams in {language} corpus (n={n})'
    plt.title(title)
    plt.savefig(png_file(title))
    plt.show()
