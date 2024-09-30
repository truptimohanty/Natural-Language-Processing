import pickle
import torch.nn as nn
from scripts.utils import get_files,convert_line2idx,convert_files2idx
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import math
torch.manual_seed(5112023)
np.random.seed(5112023)


########## All HELPER FUNCTIONS DEFINITION START ##############
def sub_seq_calc_LSTM(data,sub_length):
    """
    prepare fixed length sub sequece 
    if no of chars < 500 PAD 
    if > 500 and remainder 0 then make chunks of 500 
    if >500 and remaincer !=0 add pad in the last sequence
    """
    source = []
    target = []
    # Pad the input data if it's shorter than the target subsequence length
    if len(data) < sub_length:
        data = data + [384] * (sub_length - len(data))
        source.append(data)
        target.append((data[1:]))
        
    elif len(data) % sub_length != 0:
        data = data + [384] * (sub_length-(len(data)%sub_length))
                               
        for i in range(0,len(data),sub_length):
            source.append(data[i:i+sub_length]) 
            target.append(data[i+1:i+1+sub_length])

    elif len(data) % sub_length == 0:
        for i in range(0,len(data),sub_length):
            source.append(data[i:i+sub_length]) 
            target.append(data[i+1:i+1+sub_length])

    ## append target with 384       
    target[-1].append(384)
 
    return source,target

def flatten_matrix(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list

def default_value():
    """
    used for default dict if key not available returns 0 
    """
    return 0

def calculate_wt_vec(flat_list):
    """
    Calculating the loss weight of the vocab 
    based on freq of occurence in train data
    """
    freq = defaultdict(default_value)
    loss_weights = defaultdict(default_value)
    for l in flat_list:
        if l in freq:
            freq[l] +=1
        else:
            freq[l] =1
    total = sum([v for k,v in freq.items()])
    # print(total)
    for k in range(len(vocab)):
        loss_weights[k] = 1-(freq[k]/total)
    loss_weights_tensor = torch.tensor([v for k,v in loss_weights.items()])
    return loss_weights_tensor


def calculate_perplexity_sub(l, model):
    """
    calculate the perplexity for each sentence 
    """
    model.eval()
    with torch.no_grad():
        token_counts = len(l)
        source_lists = sub_seq_calc_LSTM(l,500)[0] 
        target_lists = sub_seq_calc_LSTM(l,500)[1]
        total_line_loss = 0
                
        hidden = (torch.zeros(num_lstm_layers,200).to(device),torch.zeros(num_lstm_layers,200).to(device)) ## intiallize hidden
                
        for s,t in zip (source_lists,target_lists): # iterate over subseq for each line 
            out,hidden = model(torch.tensor(s).to(device),hidden) # check the model output
            loss_vector = loss_fn_eval(out.view(-1, out.shape[-1]), torch.tensor(t).view(-1).to(device)) ## loss vector for each sub seq
            loss_vector = loss_vector.cpu()
            total_line_loss = total_line_loss + sum(loss_vector) ## add loss of each sub seq
        perplexity = math.e**(total_line_loss/token_counts) ##perplexity for each line
    return perplexity.numpy()
               
    
def calculate_perplexity_parallel(l2ix,model):
    """
    use the map fuction to make it fast 
    """
    model.eval()
    with torch.no_grad():
        perplexity_lines = map(lambda l_: calculate_perplexity_sub(l_, model), l2ix)
        perplexity_lines = perplexity_lines
    return list(perplexity_lines)

def generate_seq_random_new(seed_seq,seq_length,test_model):
    """
    Generate sequence with given seed sequence 
    """
    hidden = (torch.zeros(num_lstm_layers,200).to(device),torch.zeros(num_lstm_layers,200).to(device)) ## intiallize hid
    input = torch.tensor(convert_line2idx(seed_seq,vocab)).to(device)
    # print(input)
    totalchar = seed_seq
    sm = nn.Softmax(dim=1)
    for i in range(seq_length):
        out,hidden = test_model(input,hidden)
        # print('output shape',out.shape)
        ## take the softmax
        prob_out = sm(out)
        # print('prob out shape',prob_out.shape)
        chars_predict=torch.multinomial(prob_out,1)
        # print(chars_predict.shape)
        # print(chars_predict)
        next_char_index = chars_predict[-1].item()
        # print(next_char_index)
        next_char = vocab_reverse[next_char_index] 
        totalchar = totalchar + next_char
        # print(totalchar)
        input = torch.tensor([next_char_index]).to(device)
    return totalchar

########## All HELPER FUNCTIONS DEFINITION END ##############


## Define the Model 

class Char_LM(nn.Module):

    def __init__(self,input_dim = 500,demb=50, hidden_dim=200, vocab_dim=386, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_dim,demb)
        self.lstm = nn.LSTM(demb, hidden_dim, num_layers=num_layers,batch_first = True)
        self.linear1 = nn.Linear(hidden_dim,256,bias = True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256,vocab_dim,bias=True)
        
    def forward(self,x,hidden):
        # print(x.shape)
        out = self.embedding(x)
        out,hidden  = self.lstm(out,hidden)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
    
        return out,hidden


## read the vocabulary file
with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
vocab_reverse = dict((v,k) for k,v in vocab.items())


## get list of list indices from train corpus
train_l2ix = convert_files2idx(get_files('data/train'),vocab)

#first convert each line into sub sequences and flatten them to be used as batch for training 
source_train_final = torch.tensor(flatten_matrix([sub_seq_calc_LSTM(l,500)[0] for l in train_l2ix]))
target_train_final = torch.tensor(flatten_matrix([sub_seq_calc_LSTM(l,500)[1] for l in train_l2ix]))

## define the parameters 
device ='mps'      
num_lstm_layers = 2

# instantiate the model 
model = Char_LM(num_layers=num_lstm_layers).to(device)


### loss weight tensor calculation ############# 
l2ix = convert_files2idx(get_files('data/train'),vocab)
flat_list = flatten_matrix(l2ix)
loss_weights = calculate_wt_vec(flat_list)


## Define the train dataset and dataloader 
train_dataset = TensorDataset(
                            source_train_final,
                            target_train_final
                                )
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)



loss_weights = loss_weights.to(device)
lr = 0.0001 ## try with different learning rates

loss_fn = nn.CrossEntropyLoss(ignore_index=384, weight=loss_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print('Training Start') 
for epoch in tqdm(range(1)):
    train_loss = []      
    
    for source, target in train_loader:
        
        hidden = (torch.zeros(num_lstm_layers,source.size(0),200).to(device),torch.zeros(num_lstm_layers,source.size(0),200).to(device)) ## intiallize hidden

        model.train()
        
        optimizer.zero_grad()        
        out,hidden = model(source.to(device),(hidden[0].detach(),hidden[1].detach()))    #Forward pass
         
        loss = loss_fn(out.view(-1, out.shape[-1]), target.view(-1).to(device))
        loss.backward() # computing the gradients

        optimizer.step()  # Performs the optimization

        train_loss.append(loss.cpu().item())    # Appending the batch loss to the list

    save_path = f'final_trainedmodels/model_{lr}_{epoch}_2lstm_hid256.pth'
    torch.save({
            "model_param": model.state_dict(),
                },save_path)  
    
    print(f"Average training loss : {np.mean(train_loss)} after epoch {epoch}")

print('Training Over')

print('Evaluation Start')
## load the trained model 
test_model = Char_LM(num_layers=num_lstm_layers).to(device)
    
## check the file path name
checkpoint = torch.load('final_trainedmodels/model_0.0001_4_2lstm_hid256.pth')
test_model.load_state_dict(checkpoint["model_param"])
test_model.eval()

loss_fn_eval = nn.CrossEntropyLoss(reduce=False, ignore_index=384,weight=loss_weights.to(device))
     
### calculate the perplexity for dev set 
dev_l2ix = convert_files2idx(get_files('data/dev/'),vocab)
result_dev = calculate_perplexity_parallel(dev_l2ix, test_model)
perplexity_final_dev = sum(result_dev)/len(result_dev)
print('peplexity for dev set',perplexity_final_dev)


### calculate the perplexity for test set 
test_l2ix = convert_files2idx(get_files('data/test/'),vocab)
result_ = calculate_perplexity_parallel(test_l2ix, test_model)
perplexity_final = sum(result_)/len(result_)
print('peplexity for test set',perplexity_final)

print('Evaluation Ends Development and Test perplexity done')

##### Generate the sequence of next 200 characters for the seed sentence
print('Generating Sequence for next 200 Char')


print(generate_seq_random_new('The little boy was',200,test_model))
print('*************************')
print(generate_seq_random_new('Once upon a time in',200,test_model))
print('*************************')
print(generate_seq_random_new('With the target in',200,test_model))
print('*************************')
print(generate_seq_random_new('Capitals are big cities. For example,',200,test_model))
print('*************************')
print(generate_seq_random_new('A cheap alternative to',200,test_model))
