import torch
import torch.nn as nn
import numpy as np 
from tqdm import tqdm
from mp1_release.scripts.utils import get_word2ix, process_data, get_files, context_target_indices,flatten_matrix,get_eval_stats
from mp1_release.scripts.model import CBOW_MODEL
from sklearn.metrics import f1_score
import traceback
from torch.utils.data import TensorDataset, DataLoader
import argparse
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import os


def main():
    device = 'mps' ## device = 'mps' for Apple silicon  
    max_epochs = 10
    learning_rate = 0.001
    loss_fn = nn.CrossEntropyLoss()



    ## get word to index as per the vocabulary list 
    word2ix = get_word2ix(path='mp1_release/vocab.txt')

    # get the list of the train files and the dev files 
    train_files = get_files('mp1_release/data/train/')
    dev_files = get_files('mp1_release/data/dev/')

    ## get the indices for the words in the files returns list of list as per number of files 
    train_indices = process_data(train_files,5,word2ix)
    dev_indices = process_data(dev_files,5,word2ix)

    ## flatten the list of list to single list of indices 
    train_indices_list = flatten_matrix(train_indices)
    dev_indices_list= flatten_matrix(dev_indices)


    ## get the context and target indices 
    train_context_indices,train_target_indices = context_target_indices(train_indices_list,5)
    dev_context_indices,dev_target_indices = context_target_indices(dev_indices_list,5)


    # Create datasets using the TensorDataset class
    train_dataset = TensorDataset(train_context_indices,
                              train_target_indices
                              )
                        
    dev_dataset = TensorDataset(
                                dev_context_indices, 
                                dev_target_indices
                                )   
    torch.manual_seed(20)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)

    ## instantiate the model 
    model = CBOW_MODEL(18061,100).to(device)

    ## set the optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




    best_perf_dict = {"metric": 0, "epoch": 0}

    # Begin the training loop
    for epoch in tqdm(range(max_epochs)):

        train_loss = []      
        

        for inp, lab in train_loader:
            model.train()

            optimizer.zero_grad()        
            out = model(inp.to(device))    #Forward pass


            loss = loss_fn(out, lab.to(device))
            loss.backward() # computing the gradients

            optimizer.step()  # Performs the optimization

            train_loss.append(loss.cpu().item())    # Appending the batch loss to the list


            
        # Update the `best_perf_dict` if the best dev performance is beaten
        
        lab_actual = []
        lab_predict = []
        dev_loss_total = []     
        correct = 0 
        total = 0
        
        for inp_dev, lab_dev in dev_loader:
                
            model.eval()

            with torch.no_grad():

                lab_actual.extend(lab_dev.tolist()) 
                out_dev = model(inp_dev.to(device)) 

                dev_loss = loss_fn(out_dev, lab_dev.to(device))

                total = total + lab_dev.size(0)
                correct = correct + (lab_dev.to(device) == torch.argmax(out_dev,1)).sum().item()

                lab_predict.extend(torch.argmax(out_dev,1).to('cpu').tolist())

                dev_loss_total.append(dev_loss.cpu().item())
                
        Dev_F1_score = f1_score(lab_actual,lab_predict,average='macro') 

        if Dev_F1_score > best_perf_dict["metric"]:
            best_perf_dict["metric"] = Dev_F1_score
            best_perf_dict["epoch"]  = epoch

            # Saving model and optimizer parameters alongwith the F1 and epoch info
            torch.save({
                "model_param": model.state_dict(),
                "optim_param": optimizer.state_dict(),
                "dev_metric": Dev_F1_score,
                'train_loss':np.mean(train_loss),
                "dev_loss":np.mean(dev_loss_total),
                "epoch": epoch,
                "correct labels percentage": (correct/total)*100,
            }, f"./trained_models_final_submission/mean_ep{epoch}_lr{learning_rate}")    
            
        
        
        print(f"Average training loss : {np.mean(train_loss)} after epoch {epoch}")
        print(f"Average dev loss : {np.mean(dev_loss_total)} after epoch {epoch}")
        print(f'Dev data F1 score = {Dev_F1_score} after epoch number {epoch}')
        print(f"correct labels = {correct} in dev data out of total labels {total} percentage = {(correct/total)*100}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"The following exception occurred: {e}")
        print(traceback.format_exc())





