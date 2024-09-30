from typing import List, Set
from collections import deque
import torch
from sklearn.preprocessing import LabelEncoder
from torchtext.vocab import GloVe
from tqdm import tqdm
import numpy as np

device = 'mps'

encoder_classes = LabelEncoder()

glove_class = GloVe()




class Token:
    def __init__(self, idx: int, word: str, pos: str):
        self.idx = idx # Unique index of the token
        self.word = word # Token string
        self.pos  = pos # Part of speech tag
    
    def __str__(self) -> str:
        return f'id : {self.idx}, word: {self.word}, pos: {self.pos}'

class DependencyEdge:

    def __init__(self, source, target, label):
        self.source = source  # Source token index
        self.target = target  # target token index
        self.label  = label  # dependency label
        pass

    def __str__(self) -> str:
        return f'source : {self.source}, target: {self.target}, label: {self.label}'

class ParseState:
    def __init__(self, stack, parse_buffer, dependencies ):
        self.stack = stack # A stack of token indices in the sentence. Assumption: the root token has index 0, the rest of the tokens in the sentence starts with 1.
        self.parse_buffer = parse_buffer  # A buffer of token indices
        self.dependencies = dependencies
        pass

    def __str__(self) -> str:
        return f'stack : {self.stack}, buffer: {self.parse_buffer}, dependencies: {self.dependencies}'


    def add_dependency(self, source_token, target_token, label):
        self.dependencies.append(
            DependencyEdge(
                source=source_token,
                target=target_token,
                label=label,
            )
        )


def shift(ParseState) -> None:
    # TODO: Implement this as an in-place operation that updates the parse state and does not return anything
    
    ParseState.parse_buffer = deque(ParseState.parse_buffer)
    ParseState.stack.append(ParseState.parse_buffer.popleft())
    ParseState.parse_buffer = list(ParseState.parse_buffer)
    # The python documentation has some pointers on how lists can be used as stacks and queues. This may come useful:
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues



def left_arc(ParseState, label) -> None:
    # TODO: Implement this as an in-place operation that updates the parse state and does not return anything

    wi = ParseState.stack.pop()
    wj = ParseState.stack.pop()
    ParseState.stack.append(wi)
    ParseState.add_dependency(wi, wj, label)
    # ParseState.dependencies.append(label)
    # The python documentation has some pointers on how lists can be used as stacks and queues. This may come useful:
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues
    # Also, you will need to use the state.add_dependency method defined above.

    pass



def right_arc(ParseState, label) -> None:
    # TODO: Implement this as an in-place operation that updates the parse state and does not return anything

    wi = ParseState.stack.pop()
    wj = ParseState.stack.pop()
    ParseState.stack.append(wj)
    ParseState.add_dependency(wj, wi, label)

    # The python documentation has some pointers on how lists can be used as stacks and queues. This may come useful:
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues

    # Also, you will need to use the state.add_dependency method defined above.

    pass


def is_final_state(ParseState, cwindow) -> bool:
    # TODO: Implemement this
    
    # print('length of stack')
    # print(len(ParseState.stack))
    # print('length of buffer')
    # print(len(ParseState.parse_buffer))
    
    if (len(ParseState.stack) <= cwindow+1) and (len(ParseState.parse_buffer) <= cwindow):
        return True
    else:
        return False



def get_training_data_modified(words_lists, part_speech, actions, cwindow):
    """
    This is similar to the fuction provided in get_deps and modified as per the requirement. 
    Takes the word lists (list of list of str), part of speech (list of list of str), 
    actions (list str) and cwindow 
    genrates word data (from stack and buffer as per cwindow size), pos data and action data 
    """
    # Iterate over sentences
    word_data = []
    pos_data = []
    action_data = []
    
    for w_ix, words_list in enumerate(words_lists):
        # Intialize stack and buffer appropriately
        
        ## intilaize the stack with PAD as per cwindow size
        stack = [Token(idx=-i-1, word="[PAD]", pos="NULL") for i in range(cwindow)]

        parser_buff = []
        for ix in range(len(words_list)):
            parser_buff.append(Token(idx=ix, word=words_list[ix], pos=part_speech[w_ix][ix]))
        parser_buff.extend([Token(idx=ix+i+1, word="[PAD]",pos="NULL") for i in range(cwindow)])
        
        # Initilaze the parse state
        state = ParseState(stack=stack, parse_buffer=parser_buff, dependencies=[])

        # Iterate over the actions and do the necessary state changes
        for action in actions[w_ix]:
            
            # intialize train row and pos row
            train_row = []
            pos_row = []

            # stack window as per cwindow size (last two tokens)
            stack_window = state.stack[-cwindow:]

            # buffer window as per cwindow size (first two tokens)
            buffer_window = state.parse_buffer[0:cwindow]

            train_row.extend([t.word for t in stack_window]) #stack word
            train_row.extend([t.word for t in buffer_window]) #buffer word
            pos_row.extend([t.pos for t in stack_window]) #stack POS
            pos_row.extend([t.pos for t in buffer_window]) #buffer POS
            
            action_data.append(action) # action 

            word_data.append(train_row)
            pos_data.append(pos_row)

            if action == "SHIFT":
                shift(state)
                
            elif action[:8] == "REDUCE_L":
                left_arc(state, action[9:])
                     
            else:
                right_arc(state, action[9:])
                
        assert is_final_state(state,cwindow)    # Check to see that the parse is complete
        
    return word_data,pos_data,action_data


def word2_index(L,word2ix):
    """ 
    takes a of pos list and generates the indices 
    """    
    I = []
    for l in L:
       if l in word2ix.keys():
        I.append(word2ix[l])
    return I


def get_training_data_numeric(word_data, pos_data, action_data, encoder_classes, word2ix, emb_type='concat', glove_class=glove_class): 
    
    
    if emb_type == 'mean':
        word_data_numeric = [glove_class.get_vecs_by_tokens(t).mean(0) for t in word_data]

    if emb_type == 'concat':    
        word_data_numeric = [torch.flatten(glove_class.get_vecs_by_tokens(t)) for t in word_data]

    pos_data_numric = [word2_index(p,word2ix) for p in pos_data]
    
    action_data_numeric = encoder_classes.transform(action_data)
    return(torch.stack(word_data_numeric),torch.tensor(pos_data_numric),torch.tensor(action_data_numeric))


def extract_train_data_from_file(file):
    """ 
    Extract word data, pos data and actions from file
    """
    wl = []
    ps = []
    act = []

    with open(file) as f:
        for line in f.readlines():
            states = line.split('|||')
            words = states[0].split()
            wl.append(words)
            pos = states[1].split()
            ps.append(pos)
            action = states[2].split()
            act.append(action)
    
    word_data,pos_data,action_data = get_training_data_modified(wl, ps, act, 2)

    return (word_data,pos_data,action_data)


# def get_test_data_numeric(word_data,pos_data,word2ix,glove_name='840B',glove_dim=300,emb_type='concat'): 
#     ## to do according to the glove type 6B:50, 6B:300, 42B:300, 840B:300
    
#     glove = GloVe(name=glove_name, dim=glove_dim)
#     if emb_type == 'mean':
#         word_data_numeric = [glove.get_vecs_by_tokens(word_data).mean(0)]

#     if emb_type == 'concat':    
#         word_data_numeric = [torch.flatten(glove.get_vecs_by_tokens(word_data))]

#     pos_data_numric = [word2_index(pos_data,word2ix)]
    
#     return(torch.stack(word_data_numeric),torch.tensor(pos_data_numric))


def get_test_data_numeric_1(word_data,pos_data,word2ix,emb_type='mean',glove_class=glove_class): 
    """
    this converts the string word, pos to numeric and is used by predict_actions function
    """
   
    word_data_numeric = None
    
    if emb_type == 'mean':
        word_data_numeric = [glove_class.get_vecs_by_tokens(word_data).mean(0)]

    if emb_type == 'concat':    
        word_data_numeric = [torch.flatten(glove_class.get_vecs_by_tokens(word_data))]

    pos_data_numric = [word2_index(pos_data,word2ix)]
    
    return(torch.stack(word_data_numeric),torch.tensor(pos_data_numric))  



   
def predict_actions( model, words_lists, part_speech, cwindow, word2ix, emb_type='mean', encoder=encoder_classes, glove_class=glove_class, device = device):
    """  
    This is similar to the fuction provided in get_deps and modified as per the requirement. 
    Takes the trained model, word lists (list of list of str), part of speech (list of list of str), 
    actions (list str), cwindow, word2ix (pos), embeddding type (concat or mean), 
    encoder (for actions numeric to word) and glove class and generates list of list of actions. 

    """

    predict_actions = []
    
    for w_ix, words_list in enumerate(words_lists):
        # Intialize stack and buffer appropriately
        
        ## intilaize the stack with PAD as per cwindow size
        stack = [Token(idx=-i-1, word="[PAD]", pos="NULL") for i in range(cwindow)]

        parser_buff = []
        for ix in range(len(words_list)):
            # print('ix',ix)
            # print('words_list',words_list)
            parser_buff.append(Token(idx=ix, word=words_list[ix], pos=part_speech[w_ix][ix]))
        parser_buff.extend([Token(idx=ix+i+1, word="[PAD]",pos="NULL") for i in range(cwindow)])
        
        # Initialize the parse state
        state = ParseState(stack=stack, parse_buffer=parser_buff, dependencies=[])

        actions_sentence = []
        while not is_final_state(state,cwindow):
            word_row = []
            pos_row = []
            
            # stack window as per cwindow size (last two tokens)
            stack_window = state.stack[-cwindow:]

            # buffer window as per cwindow size (first two tokens)
            buffer_window = state.parse_buffer[0:cwindow]

            word_row.extend([t.word for t in stack_window]) #stack word
            word_row.extend([t.word for t in buffer_window]) #buffer word

            pos_row.extend([t.pos for t in stack_window]) #stack POS
            pos_row.extend([t.pos for t in buffer_window]) #buffer POS
        

            # xxx, yyy = get_test_data_numeric(word_row,pos_row,word2ix,glove_name='840B',glove_dim=300,emb_type='concat')
            # print(word_row)
            # print(pos_row)
            xxx, yyy = get_test_data_numeric_1(word_row, pos_row, word2ix, emb_type = emb_type, glove_class=glove_class)
            # print(xxx.shape)
            # print(yyy.shape)

            out = model(xxx.to(device), yyy.to(device)) 
            _,pred = torch.max(out,1)

            action = encoder.inverse_transform(pred.cpu())[0]
            
            # print('action before change',action)


            if len(state.stack) <= cwindow +1 :
                action = "SHIFT"

            ## checking the illegal action
            ## argsort can be used as discussed in class
            ## considering the lower probability of occurence and
            ## assuming last word as puctuation mark
            
            if len(state.parse_buffer) == cwindow and action == "SHIFT":             
                action = "REDUCE_R_punct"
    
            if action == "SHIFT":
                shift(state)
                
            if action[:8] == "REDUCE_L":
                left_arc(state, action[9:])
                        
            if action[:8] == "REDUCE_R":
                right_arc(state, action[9:])

            actions_sentence.append(action)

        predict_actions.append(actions_sentence)

    return predict_actions
