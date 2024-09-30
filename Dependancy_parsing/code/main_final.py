from typing import List, Set
from collections import deque
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
from torchtext.vocab import GloVe
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from scripts.models import PARSE_MODEL_CONCAT, PARSE_MODEL_MEAN
from scripts.state import extract_train_data_from_file,predict_actions,get_training_data_numeric
from scripts.evaluate import compute_metrics
from tqdm import tqdm
import traceback

torch.manual_seed(4102023)
np.random.seed(4102023)



def main():
    device = 'mps'
    emb_type = 'concat' ## can be used as 'mean'
    cwindow = 2

    ## ensure the model for emb_type=mean is PARSE_MODEL_MEAN
    ## ensure the model for emb_type=concat is PARSE_MODEL_CONCAT

    glove_name = '840B'
    glove_dim = 300
    glove_class = GloVe(name=glove_name, dim=glove_dim)

    ## get the word2ix dictionary matching each pos to numeric 
    df_pos = pd.read_csv('data/pos_set.txt',header=None,keep_default_na=False)
    word2ix = {k:v for v,k in enumerate(df_pos[0])}

    ## used Label encoder for muticlass actions 
    df_classes = pd.read_csv('data/tagset.txt',header=None)

    encoder = LabelEncoder()
    encoder.fit(df_classes[0])

    max_epochs = 20
    learning_rate = 0.001


    ## read the development data set used for hyper parameter tuning
    # get the list of words, pos and gold actions 

    develop_wl = []
    develop_ps = []
    develop_gold_act = []

    with open('data/test.txt') as f:
        for line in f.readlines():
            states = line.split('|||')
            develop_words = states[0].split()
            develop_wl.append(develop_words)
            develop_pos = states[1].split()
            develop_ps.append(develop_pos)
            develop_act = states[2].split()
            develop_gold_act.append(develop_act)

    ## extract the data from train file
    train_word_data, train_pos_data,train_action_data = extract_train_data_from_file('data/train.txt')

    ## convert to numeric as the dataloader works only with numeric tensor
    train_word_data_numeric, train_pos_data_numeric, train_action_data_numeric = get_training_data_numeric(train_word_data,
                                                                                                        train_pos_data,
                                                                                                        train_action_data,
                                                                                                        encoder,
                                                                                                        word2ix,
                                                                                                        emb_type=emb_type,
                                                                                                        glove_class=glove_class
                                                                                                        )

    ## extract the data from dev file 
    ## for this problem dev dataloader has not been used as we are tracking the LAS metrics
    # dev_word_data,dev_pos_data,dev_action_data = extract_train_data_from_file('data/dev.txt')
    # ## convert to numeric
    # dev_word_data_numeric, dev_pos_data_numeric, dev_action_data_numeric = get_training_data_numeric(dev_word_data,
    #                                                                                                 dev_pos_data,
    #                                                                                                 dev_action_data,
    #                                                                                                 encoder,
    #                                                                                                 word2ix,
    #                                                                                                 emb_type=emb_type,
    #                                                                                                 glove_class=glove_class
    #                                                                                                 )


    ## create train dataset 
    train_dataset = TensorDataset(train_word_data_numeric,
                                train_pos_data_numeric,
                                train_action_data_numeric
                                )

    ## create train dataloadeer                       
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # ## create dev dataset 
    # dev_dataset = TensorDataset(dev_word_data_numeric,
    #                             dev_pos_data_numeric,
    #                             dev_action_data_numeric
    #                             )

    # ## create dev dataloader                        
    # dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)


    # instantiate the model demb = embidding dimension after Glove ,dpos = pos embedding dimension 
    # h = hidden layer dimension = 200 , T = total number of actions = 75 ,P = pos vocabulary dimension = 18 

    model = PARSE_MODEL_CONCAT(demb=glove_dim,dpos=50,h=200,T=75,P=18,cwindow=cwindow).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    best_perf_dict = {"metric": 0, "epoch": 0}


    # Begin the training loop

    for epoch in tqdm(range(max_epochs)):
        train_loss = []      

        for word, pos, action in train_loader:
            model.train()

            optimizer.zero_grad()        
            out = model(word.to(device),pos.to(device))    #Forward pass


            loss = loss_fn(out, action.to(device))
            loss.backward() # computing the gradients

            optimizer.step()  # Performs the optimization

            train_loss.append(loss.cpu().item())    # Appending the batch loss to the list

        print(f"Average training loss : {np.mean(train_loss)} after epoch {epoch}")

        # predict the action for the dev data 
        develop_pred = predict_actions(model,develop_wl,develop_ps,cwindow,word2ix,emb_type=emb_type,encoder=encoder,glove_class=glove_class)

        # compute the evaluation metric LAS
        las = compute_metrics(develop_wl, develop_gold_act, develop_pred, cwindow=2)[1]
        print('DEV LAS =',las)

        if las > best_perf_dict["metric"]:
            best_perf_dict["metric"] = las
            best_perf_dict["epoch"]  = epoch

            # Saving model and optimizer parameters alongwith the train loss, dev LAS
            save_path = f'trained_models/concat_shuffle_true/model_{glove_name}_{glove_dim}_{epoch}_{learning_rate}_{emb_type}.pth'
            torch.save({
                    "model_param": model.state_dict(),
                    "optim_param": optimizer.state_dict(),
                    "dev_metric": best_perf_dict["metric"],
                    'train_loss':np.mean(train_loss),
                    "epoch": best_perf_dict["epoch"]    
                },save_path)


    ## load the trained model 
    test_model = PARSE_MODEL_CONCAT(demb=glove_dim,dpos=50,h=200,T=75,P=18,cwindow=cwindow).to(device)
    
    ## check the file path name
    checkpoint = torch.load('trained_models/concat_shuffle_true/model_840B_300_3_0.001_concat.pth')
    test_model.load_state_dict(checkpoint["model_param"])
    test_model.eval();


    ## Evaluation for the the test data 
    test_wl = []
    test_ps = []
    test_gold_act = []

    with open('data/test.txt') as f:
        for line in f.readlines():
            states = line.split('|||')
            words = states[0].split()
            test_wl.append(words)
            pos = states[1].split()
            test_ps.append(pos)
            act = states[2].split()
            test_gold_act.append(act)

    test_pred = predict_actions(test_model,test_wl,test_ps,cwindow,word2ix,emb_type=emb_type,encoder=encoder,glove_class=glove_class)

    with open('final_results.txt','w') as f:
        for l in test_pred:
            f.write(" ".join(l))
            f.write("\n")

    print('Evaluation UAS and LAS on test data',compute_metrics(test_wl, test_gold_act, test_pred, cwindow=cwindow))


    ### Qestions no #4 
    words_lists_test_1=[['Mary', 'had', 'a', 'little','lamb','.']]
    part_speech_test_1=[['PROPN', 'AUX', 'DET', 'ADJ','NOUN','PUNCT']]

    predict_actions_list_1 = predict_actions(test_model,words_lists_test_1,part_speech_test_1,cwindow,word2ix,emb_type=emb_type,
                                        encoder=encoder,glove_class=glove_class,device=device)

    print('Parsing for Marry had a little lamb .',predict_actions_list_1)

    print('*********************************************')

    words_lists_test_2=[['I','ate','the','fish','raw','.']]
    part_speech_test_2=[['PRON','VERB','DET','NOUN','ADJ','PUNCT']]

    predict_actions_list_2 = predict_actions(test_model,words_lists_test_2,part_speech_test_2,cwindow,word2ix,emb_type=emb_type,
                                        encoder=encoder,glove_class=glove_class,device=device)


    print('I ate the fish raw .',predict_actions_list_2)
    print('*********************************************')

    words_lists_test_3=[['With','neural','networks',',','I','love','solving','problems','.']]
    part_speech_test_3=[['ADP','ADJ','NOUN','PUNCT','PRON','VERB','VERB','NOUN','PUNCT']]

    predict_actions_list_3 = predict_actions(test_model,words_lists_test_3,part_speech_test_3,cwindow,word2ix,emb_type=emb_type,
                                        encoder=encoder,glove_class=glove_class,device=device)

    print('With neural networks , I love solving problems .',predict_actions_list_3)


    ## read the hidden data and save the result.
    hidden_wl = []
    hidden_ps = []
    

    with open('data/hidden.txt') as f:
        for line in f.readlines():
            states = line.split('|||')
            words = states[0].split()
            hidden_wl.append(words)
            pos = states[1].split()
            hidden_ps.append(pos)
            

    hidden_pred = predict_actions(test_model,hidden_wl,hidden_ps,cwindow,word2ix,emb_type=emb_type,encoder=encoder,glove_class=glove_class)

    # with open('results.txt','w') as f:
    #     for l in hidden_pred:
    #         f.write(" ".join(l))
    #         f.write("\n")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"The following exception occurred: {e}")
        print(traceback.format_exc())






