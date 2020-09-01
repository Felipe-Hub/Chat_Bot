import re


def clean_data(data):
    """ Takes in a list of strings (sentences) and returns a list of cleaned strings (sentences). """
    
    final_data=[]
    
    for d in data:
        if len(d.split()) > 5:
            data = d.lower().strip()
            data = data.replace('\x0c', '') 
            data = data.replace('“', '').replace('”', "")
            data = data.replace('\\', '').replace('-', '')
            data = data.replace('[', '').replace(']', '')
            data = data.replace(',', ' ,')
            data = data.replace("i'm", "i am")
            data = data.replace("'re", " are")
            data = data.replace("he's", "he is")
            data = data.replace("she's", "she is")
            data = data.replace("it's", "it is")
            data = data.replace("that's", "that is")
            data = data.replace("what's", "what is")
            data = data.replace("how's", "how is")
            data = data.replace("here's", "here is")
            data = data.replace("there's", "there is")
            data = data.replace("'ve", " have")
            data = data.replace("'d", " would")
            data = data.replace("'ll", " will")
            data = data.replace("can't", "cannot")
            data = data.replace("won't", "will not")
            data = data.replace("n't", " not")
            data = data.replace("'bout", "about")
            data = data.replace("'til", "until")
            data = data.replace("'cause", "because")
            data = data.replace("gonna", "going to")
            data = data.replace("kinda", "kind of")
            data = data.replace("n'", "ng")
            data = data.replace('"', '').replace("'", "")
            data = re.sub("[-()#/@;:<>{}`+=—\‘~|.!?]", '', data)
            data = data.replace("  ", " ")

            final_data.append(data)
        
    return final_data

def clean_entire(data):
    """ Takes in a string and returns a list of cleaned strings (sentences separated by "."). """
    data = data.lower().strip()
    data = data.replace('\x0c', ' ')
    data = data.replace('\n', ' ')
    data = data.replace('\r', ' ')
    data = data.replace('’s', '')
    data = data.replace("’", "'")
    data = data.replace('“', '.').replace('”', ".")
    data = data.replace('\\', '').replace('-', '')
    data = data.replace('[', '').replace(']', '')
    data = data.replace(',', ' , ')
    data = data.replace("i'm", "i am")
    data = data.replace("'re", " are")
    data = data.replace("he's", "he is")
    data = data.replace("she's", "she is")
    data = data.replace("it's", "it is")
    data = data.replace("that's", "that is")
    data = data.replace("what's", "what is")
    data = data.replace("how's", "how is")
    data = data.replace("here's", "here is")
    data = data.replace("there's", "there is")
    data = data.replace("'ve", " have")
    data = data.replace("'d", " would")
    data = data.replace("'ll", " will")
    data = data.replace("can't", "cannot")
    data = data.replace("won't", "will not")
    data = data.replace("n't", " not")
    data = data.replace("'bout", "about")
    data = data.replace("'til", "until")
    data = data.replace("'cause", "because")
    data = data.replace("gonna", "going to")
    data = data.replace("kinda", "kind of")
    data = data.replace("n'", "ng")
    data = data.replace('"', '').replace("'", "")
    data = data.replace('...', '.').replace('!', '.').replace('?', ".")
    #data = re.sub("[-()#/@_;:<>{}`+=—\*‘~|]", '', data)
    data = data.replace("  ", " ")
    
    # only sentences with more than 20 characters
    final_data = [' '.join(re.findall('[a-zA-Z,]+', sentence)) for sentence in data.split('.') if len(sentence) > 0]
    
    return final_data