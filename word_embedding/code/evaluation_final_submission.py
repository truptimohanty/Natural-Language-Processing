from mp1_release.scripts.utils import get_word2ix
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# load the saved model 
final_model = torch.load("trained_models_final_submission/mean_ep7_lr0.001")

# print(final_model)
## get the embeddings 
embeds = final_model['model_param']['Linear.weight'].to('cpu')

## get word to index as per the vocabulary list 
word2ix = get_word2ix(path='mp1_release/vocab.txt')


# creating an embedding_file
with open('embedding_output_final_submission.txt','w') as f:
    f.write('%s %s\n'%(18061, 100))
    for k,v in word2ix.items():
#         f.write('%s %f\n' %(k,float(str(embeds[v].detach().numpy()).strip('[').strip(']'))))
          embed_vector = " ".join([str(t) for t in embeds[v].detach().numpy()])
          f.write('%s %s\n'%(k,embed_vector))



def get_eval_stats(emb_file):   
    wv = KeyedVectors.load_word2vec_format(datapath(emb_file), binary=False)
    sim = wv.evaluate_word_pairs(os.getcwd()+"/mp1_release/test_files/wordsim_similarity_goldstandard.txt")
    analogy = wv.evaluate_word_analogies(os.getcwd()+"/mp1_release/test_files/questions-words_headered.txt")
    print(f"Word Similarity Test Pearson Correlation: {sim[0][0]}")
    print(f"Accuracy on Analogy Test: {analogy[0]}")

def get_index(word,word2ix):
        if word in word2ix.keys():
            return word2ix[word]   
        
def cos_similarity(word1,word2,embeds,word2ix):
    wv1 = embeds[get_index(word1,word2ix)]
    wv2 = embeds[get_index(word2,word2ix)]
    return nn.CosineSimilarity(dim=0)(wv1,wv2)

def analogy(k,q,m,word2ix,embeds):

    k_idx = get_index(k,word2ix)
    q_idx = get_index(q,word2ix)
    m_idx = get_index(m,word2ix)

    kv = embeds[k_idx]
    qv = embeds[q_idx]
    mv = embeds[m_idx]
   
    wv =  (qv - kv) + mv
    cosim = nn.CosineSimilarity(dim=0)
    sim_scores = [cosim(wv,embeds[v]).item() for v in range(embeds.shape[0])]
    
    value =  np.argmax(sim_scores)
    # sim_scores.sort()
    # print(value)
    
    top_5 = np.argsort(sim_scores)[::-1][:10]
    # for t in top_5:
    #     print([k for k,v in word2ix.items() if v == t])
      

    return [k for k,v in word2ix.items() if v == value]


emb_abs_path = os.path.abspath("embedding_output_final_submission.txt")
get_eval_stats(emb_abs_path)

print('cosine similarity between cat and tiger', cos_similarity('cat','tiger',embeds,word2ix))
print('cosine similarity between plane and human', cos_similarity('plane','human',embeds,word2ix))
print('cosine similarity between my and mine', cos_similarity('my','mine',embeds,word2ix))
print('cosine similarity between happy and human', cos_similarity('happy','human',embeds,word2ix))
print('cosine similarity between happy and cat', cos_similarity('happy','cat',embeds,word2ix))
print('cosine similarity between king and princess', cos_similarity('king','princess',embeds,word2ix))
print('cosine similarity between ball and racket', cos_similarity('ball','racket',embeds,word2ix))
print('cosine similarity between good and ugly', cos_similarity('good','ugly',embeds,word2ix))
print('cosine similarity between cat and racket', cos_similarity('cat','racket',embeds,word2ix))
print('cosine similarity between good and bad', cos_similarity('good','bad',embeds,word2ix))

print('analogy king:queen man:?',analogy('king','queen','man',word2ix,embeds))
print('analogy king:queen prince:?',analogy('king','queen','prince',word2ix,embeds))
print('analogy king:man queen:?',analogy('king','man','queen',word2ix,embeds))
print('analogy woman:man princess:?',analogy('woman','man','princess',word2ix,embeds))
print('analogy prince:princes man:?',analogy('prince','princess','man',word2ix,embeds))

emb_abs_path = os.path.abspath("embedding_output_final_submission.txt")
get_eval_stats(emb_abs_path)

words =  ['horse', 'cat', 'dog', 'i', 'he', 'she', 'it', 'her', 'his', 'our', 'we', 'in', 'on',
'from', 'to', 'at', 'by', 'man', 'woman', 'boy', 'girl', 'king', 'queen', 'prince',
'princess']

emb_vectors = []
for w in words:
    emb_vectors.append(embeds[get_index(w,word2ix)])

df = pd.DataFrame([tensor.tolist() for tensor in emb_vectors])

pca = PCA(n_components=2)
arr =pd.DataFrame(pca.fit_transform(df))
arr['word'] = words
print('explained variance:',pca.explained_variance_)
plt.figure(figsize=(6,6))
plt.scatter(arr[0],arr[1])
plt.title('Principal component Analysis',fontsize = 14)
plt.xlabel('Principal Component axis 1', fontsize = 14)
plt.ylabel('Principal Component axis 2',fontsize = 14)

for i in range(len(arr)):
    plt.annotate(arr['word'][i], (arr[0][i], arr[1][i] + 0.03))
plt.show()
