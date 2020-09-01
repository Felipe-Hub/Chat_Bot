import numpy as np
from pos_tagging import predict_tags
from keras.preprocessing.sequence import pad_sequences

def n_grams(X, X_tag, n=4):
    """
    This function takes a sequence of tupled sentences and tags:
    [('Here is an example', 'tag1 tag2 tag3 tag4')]
    Returns a tuple with other two tuples inside and the target (y).
    The first tuple is for the encoded texts and the second for the encoded tags.
    Each tuple contains two arrays:
    1. The first array contains padded n-grams combinations(X).
    2. The second array is composed of the next n words.
    
    All returned sequences are already pre-padded with 0.
    The maximum length of the sequences is n-1, with n being the desired n-grams.
    """  
    
    text_sequences = list()
    tags_sequences = list()
    
    for i in range(len(X)):
        text_sent_sequences = list()
        tags_sent_sequences = list()
        first_sent_words = list()
        first_sent_tags = list()
        
        for j in range(len(X[i])):
            
            first_words = X[i][:j+2]
            first_tags = X_tag[i][:j+2]
            text_sequence = X[i][j:j+n+1]
            tags_sequence = X_tag[i][j:j+n+1]
            # avoinding repeated n-grams in the same sentence
            if text_sequence not in text_sent_sequences and len(text_sequence)>1 and len(tags_sequence)>1 and len(text_sequence)<=n:
                text_sent_sequences += [text_sequence]
                tags_sent_sequences += [tags_sequence]
                first_sent_words += [first_words]
                first_sent_tags += [first_tags]

        if len(text_sent_sequences)>1:
            text_sequences += text_sent_sequences + first_sent_words
            tags_sequences += tags_sent_sequences + first_sent_tags

    print('Total Sequences: %d' % len(text_sequences))
    X = pad_sequences([seq[:-1] for seq in text_sequences], maxlen=n)
    X_tag = pad_sequences([seq[:-1] for seq in tags_sequences], maxlen=n)
    y = np.array([[seq[-1]] for seq in text_sequences])
    assert len(X) == len(y) == len(X_tag)
    
    return X, X_tag, y



def text_n_grams(X, n=4):

    text_sequences = list()
    
    for i in range(len(X)):
        text_sent_sequences = list()
        first_sent_words = list()
        
        for j in range(len(X[i])):
            
            first_words = X[i][:j+2]
            text_sequence = X[i][j:j+n+1]
            # avoinding repeated n-grams in the same sentence
            if text_sequence not in text_sent_sequences and len(text_sequence)>1 and len(text_sequence)<=n:
                text_sent_sequences += [text_sequence]
                first_sent_words += [first_words]

        if len(text_sent_sequences)>1:
            text_sequences += text_sent_sequences + first_sent_words

    print('Total Sequences: %d' % len(text_sequences))
    X = pad_sequences([seq[:-1] for seq in text_sequences], maxlen=n)
    y = np.array([[seq[-1]] for seq in text_sequences])
    assert len(X) == len(y) 
    
    return X, y


def embedded_n_grams(embedded_sent, n=4, x=100):
    """
    This function takes a sequence of embedded sentences and
    returns a tuple with four arrays. The first array contains
    vectors with padded n-grams combinations (X). The second
    array is the target embedded word (y).
    
    The third and fourth arrays are the X and y for the 
    reversed sequences (X_rev and y_rev).
    
    Both returned sequences are already pre-padded with 0.
    The maximum length of the sequences is n-1.
    The param 'x' is the size of the vector set for the embedding.
    """  
    sequences = list()
    rev_sequences = list()

    for seq in embedded_sent:

        for i in range(len(seq)):
            sent_sequences = list()
            rev_sent_sequences = list()

            for ii in range(n+1):
                sequence = seq[i:i+ii]
                rev_sequence = sequence[::-1]

                # avoinding repeated n-grams in the same sentence
                if sequence not in sent_sequences and len(sequence) > 1:
                    sent_sequences += [sequence]
                    rev_sent_sequences += [rev_sequence]

            sequences += sent_sequences
            rev_sequences += rev_sent_sequences

    assert len(sequences) == len(rev_sequences)
    print('Total Sequences: %d' % len(sequences))

    # Here we separate the previous encoded words (X) and the target
    # word we will try to predict (y), also turning them into arrays:
    X = np.array([[[0]*x]*(n-len(seq)) + seq[:-1] for seq in sequences])
    y = np.array([[seq[-1]] for seq in sequences])
    X_rev = np.array([[[0]*x]*(n-len(seq)) + seq[:-1] for seq in rev_sequences])
    y_rev = np.array([[seq[-1]] for seq in rev_sequences])

    assert len(X) == len(y)

    return X, y, X_rev, y_rev


def tagged_n_grams(tagged_sent, n=4):
    """
    This function takes a sequence of tupled sentences and tags:
    [('Here is an example', 'tag1 tag2 tag3 tag4')]
    Returns a tuple with other two tuples inside and the target (y).
    The first tuple is for the encoded texts and the second for the encoded tags.
    Each tuple contains two arrays:
    1. The first array contains padded n-grams combinations(X).
    2. The second array is composed of the next n words.
    
    All returned sequences are already pre-padded with 0.
    The maximum length of the sequences is n-1, with n being the desired n-grams.
    """  
    text_sequences = list()
    text_rev_sequences = list()
    tags_sequences = list()
    tags_rev_sequences = list()
    max_len = n - 1
    
    # Here we separate into lists of n-grams:
    for text, tag in tagged_sent:
        for i in range(len(text)):
            text_sent_sequences = list()
            text_rev_sent_sequences = list()
            tags_sent_sequences = list()
            tags_rev_sent_sequences = list()
                        
            for ii in range(n+1):
                text_sequence = text[i:i+ii]
                tags_sequence = tag[i:i+ii]
                # avoinding repeated n-grams in the same sentence
                if text_sequence not in text_sent_sequences and len(text_sequence)>1 and len(tags_sequence)>1:
                    text_rev_sequence = text[len(text_sequence):len(text_sequence)+n+1][:n-1][::-1]
                    tags_rev_sequence = tag[len(tags_sequence):len(tags_sequence)+n+1][:n-1][::-1]


                    if len(text_rev_sequence) == 0:
                        text_rev_sequence = [0]
                        tags_rev_sequence = [0]

                    text_sent_sequences += [text_sequence]
                    text_rev_sent_sequences += [text_rev_sequence]
                    tags_sent_sequences += [tags_sequence]
                    tags_rev_sent_sequences += [tags_rev_sequence]
                    assert len(text_sent_sequences) == len(tags_sent_sequences)

            if len(text_sent_sequences)>1:
                text_sequences += text_sent_sequences
                text_rev_sequences += text_rev_sent_sequences
                tags_sequences += tags_sent_sequences
                tags_rev_sequences += tags_rev_sent_sequences

    assert len(text_sequences) == len(tags_sequences) 
    print('Total Sequences: %d' % len(text_sequences))
    
    # Here we separate the previous encoded words (X) and the target
    # word we will try to predict (y), also turning them into arrays:

    X = pad_sequences([seq[:-1] for seq in text_sequences], maxlen=max_len)
    X_rev = pad_sequences([seq for seq in text_rev_sequences], padding='post', maxlen=max_len)
    
    X_tag = pad_sequences([seq[:-1] for seq in tags_sequences], maxlen=max_len)
    X_tag_rev = pad_sequences([seq for seq in tags_rev_sequences], padding='post', maxlen=max_len)
    
    y = np.array([[seq[-1]] for seq in text_sequences])
    
    assert len(X) == len(y) == len(X_tag)
    
    return ((X, X_rev),(X_tag, X_tag_rev), y)


def tagged_3D(tagged_sent, n=4):
    """
    This function takes a sequence of tupled sentences and tags:
    [('Here is an example', 'tag1 tag2 tag3 tag4')]
    Returns a tuple with other two tuples inside. The first tuple
    is for the encoded texts and the second for the encoded tags.
    Each tuple contains four arrays: the first array contains
    padded n-grams combinations(X). The second array is the target(y).
    
    The third and fourth arrays are the X and y for the reversed
    sequences (X_rev and y_rev).
    
    All returned sequences are already pre-padded with 0.
    The maximum length of the sequences is n-1, with n being the
    desired number of n-grams.
    """  
    text_sequences = list()
    text_rev_sequences = list()
    tags_sequences = list()
    tags_rev_sequences = list()
    max_len = 0
    
    # Here we separate into lists of n-grams:
    for text, tag in tagged_sent:
        if len(text) > max_len:
            max_len = len(text)
            
        for i in range(len(text)):
            text_sent_sequences = list()
            text_rev_sent_sequences = list()
            tags_sent_sequences = list()
            tags_rev_sent_sequences = list()
            
            if n == 0:
                for ii in range(len(text)):
                    text_sequence = text[i:i+ii]
                    text_rev_sequence = text_sequence[::-1]
                    tags_sequence = tag[i:i+ii]
                    tags_rev_sequence = tags_sequence[::-1]
                    
                    if text_sequence not in text_sent_sequences and len(text_sequence)>1 and len(tags_sequence)>1:
                        text_sent_sequences += [text_sequence]
                        text_rev_sent_sequences += [text_rev_sequence]
                        tags_sent_sequences += [tags_sequence]
                        tags_rev_sent_sequences += [tags_rev_sequence]
                        assert len(text_sent_sequences) == len(tags_sent_sequences)
                    
                text_sequences += text_sent_sequences
                text_rev_sequences += text_rev_sent_sequences
                tags_sequences += tags_sent_sequences
                tags_rev_sequences += tags_rev_sent_sequences

                
            else:
                for ii in range(n+1):
                    text_sequence = text[i:i+ii]
                    text_rev_sequence = text_sequence[::-1]
                    tags_sequence = tag[i:i+ii]
                    tags_rev_sequence = tags_sequence[::-1]

                    # avoinding repeated n-grams in the same sentence
                    if text_sequence not in text_sent_sequences and len(text_sequence)>1 and len(tags_sequence)>1:
                        text_sent_sequences += [text_sequence]
                        text_rev_sent_sequences += [text_rev_sequence]
                        tags_sent_sequences += [tags_sequence]
                        tags_rev_sent_sequences += [tags_rev_sequence]
                        assert len(text_sent_sequences) == len(tags_sent_sequences)

                text_sequences += text_sent_sequences
                text_rev_sequences += text_rev_sent_sequences
                tags_sequences += tags_sent_sequences
                tags_rev_sequences += tags_rev_sent_sequences

    assert len(text_sequences) == len(tags_sequences) 
    print('Total Sequences: %d' % len(text_sequences))
    
    # Here we separate the previous encoded words (X) and the target
    # word we will try to predict (y), also turning them into arrays:
    X_text = pad_sequences([seq[:-1] for seq in text_sequences], maxlen=max_len)
    X_tag = pad_sequences([seq[:-1] for seq in tags_sequences], maxlen=max_len)
    zipped = list(zip(X_text,X_tag))
    
    X = np.array(zipped)
    y = np.array([[seq[-1]] for seq in text_sequences])
   
    return X, y




def unknown_words_X(tagged_sent, n=3):
    """
    This function takes a (sentence, tags) tuple:
    ('Here is an example', 'tag1 tag2 tag3 tag4')
    
    Returns a list arrays, one array for each unknown word:
    The first vector contains padded n-grams combinations (X).
    The second vector is the reversed X (X_rev).
    The third vector is the padded n_grams for the tags (X_tag).
    The fourth vector is the reversed X_tag (X_tag_rev).
    
    All returned sequences are already pre-padded with 0.
    The maximum length of the sequences is n-1, with n being the
    desired number of n-grams.
    """
    text, tags = tagged_sent
    
    unk = [i for i, word in enumerate(text) if word=='<unk>']
    
    unknowns = []
    
    for i in unk:
        
        x = text[i+1:i+n+1][::-1]
        x_tag = tags[i+1:i+n+1][::-1]
        x_rev = text[::-1][i+1:i+n][::-1]
        x_tag_rev = tags[::-1][i+1:i+n][::-1]
        
        for pos,num in enumerate(x):
            if num=='<unk>':
                x[pos] = 0
        for pos,num in enumerate(x_tag):
            if num=='<unk>':
                x_tag[pos] = 0
        for pos,num in enumerate(x_rev):
            if num=='<unk>':
                x_rev[pos] = 0
        for pos,num in enumerate(x_tag_rev):
            if num=='<unk>':
                x_tag_rev[pos] = 0
            
        X = [[0]*(n-len(x)) + x]
        X_tag = [[0]*(n-len(x_tag)) + x_tag]
        X_rev = [[0]*(n-len(x_rev)) + x_rev]
        X_tag_rev = [[0]*(n-len(x_tag_rev)) + x_tag_rev]
        
        unknowns += [[np.array(X), np.array(X_tag), np.array(X_rev), np.array(X_tag_rev)]]
        
    return unknowns


def check_and_predict(sent, tk_text, tk_tags, hmm_model, oov_model, max_length): 
    """
    This function takes a string and returns it encoded and pos-tagged on a tuple.
    Also, verifies if there are unknown words in the sentence, replacing unknown
    words for the word predicted by the 'oov_model'.
    """
    # predict and encode preliminary tags
    tags = predict_tags([sent], hmm_model)['tagged'][0][1]
    dec_tags = tk_tags.texts_to_sequences([tags])
    
    checked_sentence = list()
    unknown_words = list()
    # check for unknown words in the vocabulary
    for i, word in enumerate(sent.split()):
        if word not in tk_text.word_index:
            checked_sentence.append('<unk>')
            unknown_words.append(word)
        else:
            checked_sentence.append(tk_text.word_index[word])
    print('Encoded sentence:', checked_sentence)    
    
    # if there is no unknown words, return tagged and encoded sentences
    if '<unk>' not in checked_sentence:
        print("You do not need to go further, all words are in vocabulary!")
        return (checked_sentence, dec_tags[0])
    
    # check if there are enough known words for good prediction
    elif len(unknown_words)+1 >= len(checked_sentence)/2:
        print("Sorry, don't know what you mean with:", ' '.join(unknown_words))
        return (None,None)
    
    # else, predict the word and return tagged and encoded sentences
    else:
        
        # reversed dictionary to find word by their sequence code
        seq2word_map = dict(map(reversed, tk_text.word_index.items()))

        # keeping the same n-grams as trained data
        n = max_length

        # prepare input for model (X)
        tagged = (checked_sentence, dec_tags[0])
        # X will be the sequence of n words before and after the unknown word
        X = unknown_words_X(tagged, n)
       
        pred_position = list()
        # calculate the word probabilities
        for i in X:
            prob_vec = oov_model.predict(i)
            y_hat = np.argmax(prob_vec) # get index with highest probability
            pred_position.append(y_hat)
        
        predicted = list()
        j = 0
        # replace unknown words for the word in the vocabulary with highest similarity 
        for unk in checked_sentence:
            if unk == '<unk>':
                word = seq2word_map[pred_position[j]]
                j+=1
            else:
                word = seq2word_map[unk]
            predicted.append(word)
        predicted = ' '.join(predicted)
        print('Sentence with replaced unknown words:', predicted[5:-5])
        # predict actual tags for replaced words
        tags = predict_tags([predicted], hmm_model)['tagged'][0][1]
        dec_sentence = tk_text.texts_to_sequences([predicted])
        dec_tags = tk_tags.texts_to_sequences([tags])

        return (dec_sentence[0], dec_tags[0])