import re
import numpy as np
import pandas as pd
import pickle
import json
import keras

from data_cleaning import clean_data, clean_entire
from oov_prep import tagged_n_grams, unknown_words_X, check_and_predict, tagged_3D
from pos_tagging import predict_tags, corpus

def little_helpers():
    print('potions: re, numpy as np, pandas as pd, pickle, json, nltk, keras, collections')
    print('spells: clean_data, clean_entire, predict_tags, tagged_3D, tagged_n_grams, unknown_words_X, check_and_predict')
    pass
    
   
