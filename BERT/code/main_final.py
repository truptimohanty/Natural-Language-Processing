from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


torch.manual_seed(5112023)
np.random.seed(5112023)

device = 'mps'
num_epochs = 10

### load datasets #########
dataset_rte = load_dataset('yangwang825/rte')
dataset_sst2 = load_dataset('gpt3mix/sst2')

###### define the path name ##########
model_path_bert_tiny = 'prajjwal1/bert-tiny'
model_path_bert_mini = 'prajjwal1/bert-mini'

## define custom data loader 
def custom_dataloader(dataset,model_path,task = 'rte',max_length=512,shuffle=False):
    """
    inputs : dataset, model path, task = rte or sst 
    output : dataloader with batch
    """
    tokenizer= BertTokenizer.from_pretrained(model_path)
    if task =='rte':
        source_train = tokenizer(
                                dataset['text1'],
                                dataset['text2'],
                                max_length=max_length,
                                truncation=True, 
                                padding=True, 
                                return_tensors='pt')
    elif task =='sst':
        
        source_train= tokenizer(
                                dataset['text'],
                                max_length=max_length,
                                truncation=True, 
                                padding=True, 
                                return_tensors='pt')

    
    target_train = torch.tensor(dataset['label'], dtype = torch.long)

    train_dataset = TensorDataset(
                                source_train['input_ids'],
                                source_train['token_type_ids'],
                                source_train['attention_mask'],
                                target_train
                                    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=shuffle)
    return train_loader


#### Define the class ################
class model_classification(nn.Module):
    def __init__(self,model_path,hidden_dim, class_labels, fine_tune_bert=False):
        super().__init__()
        
        self.bert_model = BertModel.from_pretrained(model_path).to(device)
        
        ## fine tune true or false
        self.fine_tune_bert = fine_tune_bert
        ## if fine_tune_bert=False then bert model requires_grad = false
        self.fine_tune_function()
        ## linear
        self.Linear1 = nn.Linear(hidden_dim, class_labels)


    def forward(self,input_ids,token_type_ids,attention_mask):
        
        # use the cls o/p from bert model
        cls = self.bert_model(input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask).pooler_output
        # pass it to linear model
        out = self.Linear1(cls)
        return out


    def fine_tune_function(self): ## if fine_tune_bert=False then bert model requires_grad = false
        for name,param in self.bert_model.named_parameters():
            param.requires_grad = self.fine_tune_bert

########### define train_model function ##################
def train_model(model,train_loader,optimizer,loss_fn):
    train_loss = []    
    
    for input_ids,token_type_ids,attention_mask,target in train_loader:
        input_ids = input_ids.to(device)  
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        target = target.to(device)
        model.train()
        optimizer.zero_grad()  
        out = model(input_ids,token_type_ids,attention_mask)
        loss = loss_fn(out,target)
        # print(loss)   
        loss.backward() # computing the gradients
        optimizer.step()  # Performs the optimization
        train_loss.append(loss.cpu().item())    
    print(f"Average training loss : {np.mean(train_loss)}")


########### define eval_model function ##################
def eval_model(model,test_loader,loss_fn):
    """
    evaluate the loss, accuracy, F1 score 
    """
    correct = 0
    total = 0
    target_actual = []
    target_predict = []
    dev_loss_total = []
    model.eval()
    with torch.no_grad():
        for input_ids,token_type_ids,attention_mask,target in test_loader:
            target_actual.extend(target.tolist()) 
        
            input_ids = input_ids.to(device)
           
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            target = target.to(device)
            out = model(input_ids,token_type_ids,attention_mask)
            loss = loss_fn(out,target)
            predict = torch.argmax(out,1)

            total = total + target.size(0)
            correct = correct + (target == predict).sum().item()

            target_predict.extend(predict.to('cpu').tolist())

            dev_loss_total.append(loss.cpu().item())
                
        dev_F1_score = f1_score(target_actual,target_predict,average='macro') 
        accuracy = correct/total
        dev_loss = np.mean(dev_loss_total)
        print('dev loss and dev accuracy',dev_loss,accuracy)
    return dev_loss, accuracy, dev_F1_score,target_actual,target_predict

##### test_model for the hidden dataset ##########
def test_model_hidden(best_model,model_path=model_path_bert_mini,infile=None,outfile=None,task ='rte'):
    """
    read the csv file and write the prediction and proabability into csv file
    """
    tokenizer=BertTokenizer.from_pretrained(model_path)
    df = pd.read_csv(infile)

    if task =='rte':
        hidden_dataset = {}
        hidden_dataset['text1'] = df['text1'].values.tolist()
        hidden_dataset['text2'] = df['text2'].values.tolist()

        test_tokens = tokenizer(
                                hidden_dataset['text1'],
                                hidden_dataset['text2'],
                                max_length=512,
                                truncation=True, 
                                padding=True, 
                                return_tensors='pt')
    elif task =='sst':
        hidden_dataset = {}
        hidden_dataset['text'] = df['text'].values.tolist()
 

        test_tokens = tokenizer(
                                hidden_dataset['text'],
                               
                                max_length=512,
                                truncation=True, 
                                padding=True, 
                                return_tensors='pt')

    output = best_model (              
                                test_tokens['input_ids'].to(device),
                                test_tokens['token_type_ids'].to(device),
                                test_tokens['attention_mask'].to(device)
                                )
    
    softmax_scores = nn.functional.softmax(output,dim=1)
    # print(softmax_scores)
    prob,predict = torch.max(softmax_scores,dim=1)
    prob_0 = softmax_scores[:,0]
    prob_1 = softmax_scores[:,1]
    # prob_min,_ = torch.min(softmax_scores,dim=1)
    df['predict'] = predict.detach().to('cpu').numpy()
    df['correct_class_prob'] = prob.detach().to('cpu').numpy()
    df['probab_0'] = prob_0.detach().to('cpu').numpy()
    df['probab_1'] = prob_1.detach().to('cpu').numpy()
    # df['prob_other'] = prob_min.detach().to('cpu').numpy()
    df.to_csv(outfile)
    return df

###### create train and dev dataloader for differnent tasks #######

train_loader = custom_dataloader(dataset_sst2['train'],
                                          model_path = model_path_bert_mini,
                                          task = 'sst',
                                          shuffle=True)

dev_loader = custom_dataloader(dataset_sst2['validation'],
                                        model_path = model_path_bert_mini,
                                        task = 'sst',                 
                                        shuffle=False)

# train_loader = custom_dataloader(dataset_rte['train'],
#                                           model_path = model_path_bert_mini,
#                                           task = 'rte',
#                                           shuffle=True)

# dev_loader = custom_dataloader(dataset_rte['validation'],
#                                         model_path = model_path_bert_mini,
#                                         task = 'rte',                 
#                                         shuffle=False)


###### start training and save the best model based on dev accuracy #############
loss_fn = nn.CrossEntropyLoss()
dev_accuracy = 0
for lr in [1e-4,1e-5,1e-6]:
    model = model_classification(model_path_bert_mini,256,2,fine_tune_bert=True).to(device)
    # model = model_classification(model_path_bert_tiny,128,2,fine_tune_bert=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

    for epoch in tqdm(range(10)):
        train_model(model,train_loader,optimizer,loss_fn)
        dev_loss, accuracy, dev_F1_score,target_actual,target_predict = eval_model(model,dev_loader,loss_fn)
        if accuracy > dev_accuracy:
            dev_accuracy = accuracy
            save_path = f'trained_models/model_{lr}_{epoch}_sst_{dev_accuracy}_mini_ft.pth'
            torch.save({
                "model_param": model.state_dict(),
                    },save_path)  




########### Evaluate for the test dataset plot confusion marix ####################
test_loader = custom_dataloader(dataset_sst2['test'],
                                        model_path = model_path_bert_mini,
                                        task = 'sst',                 
                                        shuffle=False)

loss_fn = nn.CrossEntropyLoss()
## load the best model 
best_model  = model_classification(model_path_bert_mini,256,2,fine_tune_bert=True).to(device)

## check the file path name
checkpoint = torch.load('trained_models/Final_models/model_0.0001_4_sst_0.8314220183486238_mini_ft.pth')
best_model.load_state_dict(checkpoint["model_param"])

test_loss, accuracy, test_F1_score,target_actual,target_predict = eval_model(best_model,test_loader,loss_fn)
print('test accuracy',accuracy)
print('test F1 score',test_F1_score)
cm = confusion_matrix(target_actual, target_predict, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[0,1])
plt.figure(figsize=(2,2))
disp.plot(cmap = plt.cm.Blues)
plt.title(f'SST: BERT Mini TEST DATA With Fine Tuning')

plt.show()



### Generate prediction and probabilities for the hidden data 

########## FOR SST ###########################
best_model  = model_classification(model_path_bert_mini,256,2,fine_tune_bert=True).to(device)

## check the file path name
checkpoint = torch.load('trained_models/Final_models/model_0.0001_4_sst_0.8314220183486238_mini_ft.pth')
best_model.load_state_dict(checkpoint["model_param"])

test_model_hidden(best_model, model_path=model_path_bert_mini, infile="hidden_sst2.csv",outfile="results_sst.csv",task ='sst')

########## FOR RTE ###########################
best_model  = model_classification(model_path_bert_mini,256,2,fine_tune_bert=True).to(device)

## check the file path name
checkpoint = torch.load('trained_models/Final_models/model_0.0001_4_rte_0.6498194945848376_mini_ft.pth')
best_model.load_state_dict(checkpoint["model_param"])

test_model_hidden(best_model, model_path=model_path_bert_mini, infile="hidden_rte.csv",outfile="results_rte.csv",task ='rte')

######### Define a random classifier ############
def model_random_classifier(dataset,task='rte'):
    if task =='rte':
        data_size = len(dataset['text1'])
    elif task =='sst':
        data_size = len(dataset['text'])
    return torch.randint(0,2,(data_size,),dtype=torch.int32)

print('predict for RTE test data using random classifier')
predict_test_rte = model_random_classifier(dataset_rte['test'],task='rte')
print('accuracy with random classifier',accuracy_score(dataset_rte['test']['label'],predict_test_rte))

print('####################')
print('predict for SST test data using random classifier')
predict_test_rte = model_random_classifier(dataset_sst2['test'],task='sst')
print('accuracy with random classifier',accuracy_score(dataset_sst2['test']['label'],predict_test_rte))

## Q4 FOR THE GIVEN DATA IN THE MP4 PDF create a csv file mp4_sst.csv and mp4_rte.csv
# RTE
## load the best model 
best_model  = model_classification(model_path_bert_mini,256,2,fine_tune_bert=True).to(device)

## check the file path name
checkpoint = torch.load('trained_models/Final_models/model_0.0001_4_rte_0.6498194945848376_mini_ft.pth')
best_model.load_state_dict(checkpoint["model_param"])

print(test_model_hidden(best_model, model_path=model_path_bert_mini, infile="mp4_rte.csv",outfile="results_mp4_rte.csv",task ='rte'))

# SST
## load the best model 
best_model  = model_classification(model_path_bert_mini,256,2,fine_tune_bert=True).to(device)

## check the file path name
checkpoint = torch.load('trained_models/Final_models/model_0.0001_4_sst_0.8314220183486238_mini_ft.pth')
best_model.load_state_dict(checkpoint["model_param"])

print(test_model_hidden(best_model, model_path=model_path_bert_mini, infile="mp4_sst.csv",outfile="results_mp4_sst.csv",task ='sst'))