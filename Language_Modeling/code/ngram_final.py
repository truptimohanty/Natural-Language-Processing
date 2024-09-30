import pickle
from scripts.utils import get_files,convert_line2idx,convert_files2idx
from collections import defaultdict
import numpy as np 


### get the vocab dictionary from the pickle file
with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)


###### HELPER FUNCTIONS ##############

def sub_seq_calc(data,sub_length):
    '''
    create ngrams depending on sub_length
    first pad the start of the data 
    ouput: get the subsequences 
    '''
    source = []
    data_m = [384] * (sub_length - 1) + data
                       
    for i in range(0,len(data)):
        source.append(data_m[i:i+sub_length])
         
    return source



def sub_seq_calc_oneless(data,sub_length):
    """
    from n grams of sub_length size obtain the n-1 grams
    """
    source = []
    data_m = [384] * (sub_length - 1) + data
                       
    for i in range(0,len(data)+1):
        source.append(data_m[i:i+sub_length-1])
         
    return source


def flatten_matrix(matrix):
    """
    Flatten the matrix 
    """
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list

def default_value():
    """
    creating a default dict with default value 0.
    """
    return 0

def feeqcounts(list):
    """
    get the counts for 4grams
    """
    freq = defaultdict(default_value)
    for l in list:
        item = tuple(l)
        if item in freq:
            freq[item] +=1
        else:
            freq[item] =1
    return freq


def calculate_perplexity_sentence(sentence,freq_fourgram,freq_threegram):
    """
    Caluculate the peplexity for each sentence
    """
    N = len(sentence)
    # print(N)
    sum = 0
    for fourgram in sentence:
        # print('four',freq_fourgram[tuple(fourgram)])
        # print('three',freq_threegram[tuple(fourgram[0:3])])
        prob = (freq_fourgram[tuple(fourgram)]+1)/(freq_threegram[tuple(fourgram[0:3])]+386)
        # print(prob)
        sum = sum + np.log2(prob)

    perplexity = 2**((-1/N)*(sum))
    return perplexity

def calculate_perplexity_corpus(corpus,freq_fourgram,freq_threegram):
    """
    Caluculate the peplexity for corpus
    """
    total_sentence = len(corpus)
    sum = 0
    for c in corpus:
        sum = sum + calculate_perplexity_sentence(c,freq_fourgram,freq_threegram)
    perplexity = sum/total_sentence
    return perplexity

print('start the program')
## get list of list indices from train corpus
l2ix = convert_files2idx(get_files('data/train'),vocab)

flat_fourgram = flatten_matrix([sub_seq_calc(l,4) for l in l2ix])
print('counts the four gram occurences') 
freq_fourgram = feeqcounts(flat_fourgram)

flat_threegram = flatten_matrix([sub_seq_calc_oneless(l,4) for l in l2ix])
print('calculate the three gram occurences')
freq_threegram = feeqcounts(flat_threegram)


### ## get list of list indices from test corpus
l2ix = convert_files2idx(get_files('data/test'),vocab)
fourgram_test = [sub_seq_calc(l,4) for l in l2ix]

peplexity_corpus = calculate_perplexity_corpus(fourgram_test,freq_fourgram,freq_threegram) 
print('Perplexity of test corpus',peplexity_corpus)
print('total parameters',len(freq_fourgram) + len(freq_threegram))