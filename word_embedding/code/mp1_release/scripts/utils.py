import glob
import torch.nn as nn
import torch

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import os


def get_word2ix(path = "./../vocab.txt"):
    """ Generates a mapping given a vocabulary file. 
    Input
    -------------
    path: str or pathlib.Path. Relative path to the vocabulary file. 

    Output
    -------------
    word2ix: dict. Dictionary mapping words to unique IDs. Keys are words and 
                values are the indices.
    """
    word2ix = {}
    with open(path) as f:
        data = f.readlines()
        for line in data:
            word2ix[line.split("\t")[1].strip()] = int(line.split("\t")[0])
    
    return word2ix



def get_files(path):
    """ Returns a list of text files in the 'path' directory.
    Input
    ------------
    path: str or pathlib.Path. Directory path to load files from. 

    Output
    -----------
    file_list: List. List of paths to text files
    """
    file_list =  list(glob.glob(f"{path}/*.txt"))
    return file_list


def process_data(files, context_window, word2ix):
    """ Returns the processed data. Processing involves reading data from
    the files, converting the words to appropriate indices, mapping OOV words
    to the [UNK] token and padding appropriately.
    Inputs
    -----------
    files: List. List of files to be processed. Can be the list
            returned by the `get_files()` method.
    context_window: int. Size of the context window. Size is the amount
            of words considered as context either to the left or right of a word
    word2ix: dict. Mapping from word to a unique index. Can be the dict returned by
                the `get_word2ix` method

    Output
    ----------
    data: List[List[int]]. Each list corresponds to a file and the set of indices
            for the contents of the file.
    """
    data = []
    for file in files:
        file_data = [word2ix["[PAD]"]]*context_window
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                if line.strip() not in word2ix.keys():
                    file_data.append(word2ix["[UNK]"])
                else:
                    file_data.append(word2ix[line.strip()])
            
            file_data.extend([word2ix["[PAD]"]]*context_window)
            data.append(file_data.copy())

    return data


def context_target_indices(L,context_size = 5, padding_id = 18060):
    ## i have used the padding id as 18060 in vocab.txt file 
    i = 0
    context_indices = []
    target_indices = []
    while (i < len(L)):
        if L[i] == padding_id:
            i = i+1
            continue
        target_indices.append(L[i])
        context_indices.append(L[i-context_size:i] + L[i+1:i+context_size+1])
        i = i+1
    context_indices = torch.tensor(context_indices,dtype=torch.int64)
    target_indices = torch.tensor(target_indices,dtype=torch.int64)
    return context_indices,target_indices


def flatten_matrix(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list




# class CBOW(nn.Module):
    
#     def __init__(self,V_size,E_size):
#         super().__init__()
#         self.Embedding = nn.Embedding(V_size,E_size)
#         self.Linear = nn.Linear(E_size,V_size,bias=False)
        
#     def forward(self,input):
# #         print('input shape',input.shape)
#         h = self.Embedding(input)
# #         print('after embedding shape',h.shape)
# #         print(h)
#         h = h.sum(1)
# #         print('after sum shape',h.shape)
# #         print(h)
#         out = self.Linear(h)
# #         print('after classification',out.shape)
# #         print(out)
#         return(out)


def get_eval_stats(emb_file):   
    wv = KeyedVectors.load_word2vec_format(datapath(emb_file), binary=False)
    sim_abs_path = os.path.abspath('mp1_release/test_files/wordsim_similarity_goldstandard.txt')
    sim = wv.evaluate_word_pairs(sim_abs_path)
    analogy_abs_path = os.path.abspath("mp1_release/test_files/questions-words_headered.txt")
    analogy = wv.evaluate_word_analogies(analogy_abs_path)
    print(f"Word Similarity Test Pearson Correlation: {sim[0][0]}")
    print(f"Accuracy on Analogy Test: {analogy[0]}")
