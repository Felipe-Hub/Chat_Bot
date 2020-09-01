import matplotlib.pyplot as plt
import numpy as np

from itertools import chain
from collections import Counter, defaultdict
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution

from nltk import pos_tag, word_tokenize
from nltk.corpus import brown
from sklearn.model_selection import train_test_split

from nltk.corpus import brown
import nltk
nltk.download('brown')

# Define corpus

corpus = brown.tagged_sents()

training_vocab = list(set([word for sent in corpus for word, tag in sent]))


def pair_counts(X, Y):
    """Return a dictionary keyed to each unique value in the first sequence list
    that counts the number of occurrences of the corresponding value from the
    second sequences list.
    """
    words = [ii for i in X for ii in i if type(i) != str]
    tags = [ii for i in Y for ii in i if type(i) != str]

    pair_count = {tag:{} for tag in set(tags)}
    
    for tag, word in zip(tags, words):
        pair_count[tag][word] = pair_count[tag].get(word, 0) + 1
    
    return pair_count


def unigram_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequence list that
    counts the number of occurrences of the value in the sequences list. The sequences
    collection should be a 2-dimensional array.
    """
    tags = [ii for i in sequences for ii in i if type(i) != str]
    
    unigram_counts = {tag:tags.count(tag) for tag in set(tags)}
    
    return unigram_counts


def bigram_counts(sequences):
    """Return a dictionary keyed to each unique PAIR of values in the input sequences
    list that counts the number of occurrences of pair in the sequences list. The input
    should be a 2-dimensional array.
    """
    
    bigram_counts = Counter()

    for sequence in sequences:
        for tag1, tag2 in zip(sequence[:-1], sequence[1:]):
            bigram_counts[(tag1, tag2)] += 1
    
    return bigram_counts


def starting_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the beginning of
    a sequence.
    """
    
    init_tags = [sentence[0] for sentence in sequences]
    starting_counts = Counter(init_tags)
    
    return starting_counts


def ending_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the end of
    a sequence.
    """
    end_tags = [sentence[-1] for sentence in sequences]
    ending_counts = Counter(end_tags)
    
    return ending_counts


# For accuracy testing

def replace_unknown(sequence, vocab=training_vocab):
    """Return a copy of the input sequence where each unknown word is replaced
    by the literal string value 'nan'. Pomegranate will ignore these values
    during computation.
    """
    return [w if w in vocab else 'nan' for w in sequence]



def predict_tags(data, model):
    """ Takes in a sentence in a string and returns a tuple
    (list of words, list of tags)  
    """
    tags = list()
    text = data.split()

    try:
        _, state_path = model.viterbi(replace_unknown(text))
    except:
        state_path = None

    if state_path != None:
        pred_tags = [state[1].name for state in state_path[1:-1]]
        tags.append(pred_tags)

    else:
        tags.append('UNK')

    return text, tags[0]



def my_accuracy(X, Y, model, training_vocab):
    """Calculate the prediction accuracy by using the model to decode each sequence
    in the input X and comparing the prediction with the true labels in Y.
    """
    correct = total_predictions = 0
    for observations, actual_tags in zip(X, Y):
        
        # The model.viterbi call in simplify_decoding will return None if the HMM raises an error.
        # Any exception counts the full sentence as an error.
        try:
            most_likely_tags = my_simplify_decoding(observations, model, training_vocab)
            correct += sum(p == t for p, t in zip(most_likely_tags, actual_tags))
        except:
            pass
        total_predictions += len(observations)
    return correct / total_predictions
