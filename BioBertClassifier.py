# -*- coding: utf-8 -*-


import torch
import numpy as np
import time
import math
import datetime
import torch.nn as nn
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer







def biobertclassifier(rawdata,model_name):
# --------------------------------
#  Transformer parameters
# --------------------------------
    max_seq_length = 256  
    batch_size = 100  # normally 100

# --------------------------------
#  GAN-BERT specific parameters
# --------------------------------
# number of hidden layers in the generator,
# each of the size of the output space
    num_hidden_layers_g = 1
# number of hidden layers in the discriminator,
# each of the size of the input space
    num_hidden_layers_d = 1
# size of the generator's input noisy vectors
    noise_size = 100
# dropout to be applied to discriminator's input vectors
    out_dropout_rate = 0.2

# Replicate labeled data to balance poorly represented datasets,
# e.g., less than 1% of labeled material
    apply_balance = False

    if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
    else:
        device = torch.device("cpu")


    transformer = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

# This label list needs to match the list used when training the model 
    label_list = ["UNK_UNK","Positive","Negative","Neutral"]

    class Generator(nn.Module):
        def __init__(self, noise_size=100, output_size=512, hidden_sizes=[512], dropout_rate=0.1):
            super(Generator, self).__init__()
            layers = []
            hidden_sizes = [noise_size] + hidden_sizes
            for i in range(len(hidden_sizes)-1):
                layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

            layers.append(nn.Linear(hidden_sizes[-1], output_size))
            self.layers = nn.Sequential(*layers)

        def forward(self, noise):
            output_rep = self.layers(noise)
            return output_rep

# ------------------------------
#   The Discriminator
#   https://www.aclweb.org/anthology/2020.acl-main.191/
#   https://github.com/crux82/ganbert
# ------------------------------


    class Discriminator(nn.Module):
        def __init__(self, input_size=512, hidden_sizes=[512], num_labels=2, dropout_rate=0.1):
            super(Discriminator, self).__init__()
            self.input_dropout = nn.Dropout(p=dropout_rate)
            layers = []
            hidden_sizes = [input_size] + hidden_sizes
            for i in range(len(hidden_sizes)-1):
                layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

            self.layers = nn.Sequential(*layers)  # per il flatten
            self.logit = nn.Linear(hidden_sizes[-1], num_labels+1)
            # +1 for the probability of this sample being fake/real.
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, input_rep):
            input_rep = self.input_dropout(input_rep)
            last_rep = self.layers(input_rep)
            logits = self.logit(last_rep)
            probs = self.softmax(logits)
            return last_rep, logits, probs

 
    hidden_size = int(config.hidden_size)
    # Define the number and width of hidden layers
    hidden_levels_g = [hidden_size for i in range(0, num_hidden_layers_g)]
    hidden_levels_d = [hidden_size for i in range(0, num_hidden_layers_d)]



    generator = Generator(noise_size=noise_size, output_size=hidden_size, hidden_sizes=hidden_levels_g, dropout_rate=out_dropout_rate)
    discriminator = Discriminator(input_size=hidden_size, hidden_sizes=hidden_levels_d, num_labels=len(label_list), dropout_rate=out_dropout_rate)
    genloc = model_name + '/SavedGenerator.pt'
    disloc = model_name + '/SavedDiscriminator.pt'
    
    generator.load_state_dict(torch.load(genloc))
    discriminator.load_state_dict(torch.load(disloc))

    
    def abstractprep(rawdata):
        UnlabeledDataRaw = rawdata
        
        def conclusionnum(abstracttext):
            cnum = None
            for i in range(0, len(abstracttext)-1):
                if abstracttext[i].startswith('CONCL') == 1:
                    cnum = i
                if abstracttext[i].startswith('CLINICAL TRIAL REGISTRATION:'):
                    abstracttext.pop(i)
                if cnum == None:
                    cnum = math.ceil(len(abstracttext)*0.125)
                    cnum = len(abstracttext)-cnum-1
            return cnum


        def abstractconclusionpuller(data):
            abstracts = []
            for pap in range(0, len(data)):
                abstract = data[pap]
                abstracttext = nltk.tokenize.sent_tokenize(abstract)
                cnum = conclusionnum(abstracttext)
                abstracttext = TreebankWordDetokenizer().detokenize(abstracttext[cnum:len(abstracttext)])
                abstracts.append(abstracttext)
            return abstracts


        UnlabeledAbstracts = abstractconclusionpuller(UnlabeledDataRaw)


        def abstracttupler(abstracts, labels):
            '''Convert Score to Pos Neg Neutral or
            unknown and creates the input list'''
            tuplelist = []
            for abstract in range(0, len(abstracts)):
                label = labels[abstract]      
                if label == 1:
                    label = "Positive"
                elif label == 0:
                    label = "Neutral"
                elif label == -1:
                    label = "Negative"
                else:
                    label = "UNK_UNK"
                abstuple = (abstracts[abstract], str(label))
                tuplelist.append(abstuple)
            return tuplelist


        nullclasses = [4] * len(UnlabeledAbstracts) # Uses a false label of "4' to crate the UNK class
        UnlabeledData = abstracttupler(UnlabeledAbstracts, nullclasses)

        return UnlabeledData, UnlabeledDataRaw


    def generate_data_loader(input_examples, label_masks, label_map, batch_size, train, do_shuffle = False, balance_label_examples = False):
        '''
        Generate a Dataloader given the input examples, eventually masked if they are 
        to be considered NOT labeled.
        '''
        examples = []

        # Count the percentage of labeled examples  
        num_labeled_examples = 0
        for label_mask in label_masks:
            if label_mask: 
                num_labeled_examples += 1
                label_mask_rate = num_labeled_examples/len(input_examples)

  # if required it applies the balance
        for index, ex in enumerate(input_examples): 
            if label_mask_rate == 1 or not balance_label_examples:
                examples.append((ex, label_masks[index]))
            else:
        # IT SIMULATE A LABELED EXAMPLE
                if label_masks[index]:
                    balance = int(1/label_mask_rate)
                    balance = int(math.log(balance,2))
                    if balance < 1:
                      balance = 1
                    for b in range(0, int(balance)):
                      examples.append((ex, label_masks[index]))
                else:
                    examples.append((ex, label_masks[index]))
          
  #-----------------------------------------------
  # Generate input examples to the Transformer
  #-----------------------------------------------
        input_ids = []
        input_mask_array = []
        label_mask_array = []
        label_id_array = []

  # Tokenization 
        for (text, label_mask) in examples:
            encoded_sent = tokenizer.encode(text[0], add_special_tokens=True, max_length=max_seq_length, padding="max_length", truncation=True)
            input_ids.append(encoded_sent)
            label_id_array.append(label_map[text[1]])
            label_mask_array.append(label_mask)
  

  # Attention to token (to ignore padded input wordpieces)
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]                          
            input_mask_array.append(att_mask)
  
  # Conversion to Tensor
        input_ids = torch.tensor(input_ids) 
        input_mask_array = torch.tensor(input_mask_array)
        label_id_array = torch.tensor(label_id_array, dtype=torch.long)
        label_mask_array = torch.tensor(label_mask_array)

  # Building the TensorDataset

        dataset = TensorDataset(input_ids, input_mask_array, label_id_array, label_mask_array)

        if do_shuffle:
            sampler = RandomSampler
        else:
            sampler = SequentialSampler

# Building the DataLoader
        return DataLoader(
                  dataset,  # The training samples.
                  sampler=sampler(dataset),
                  batch_size=batch_size)  # Trains with this batch size.


    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    unlabeled_examples, unlabeled_examplesRaw = abstractprep(rawdata)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    classifier_label_masks = np.ones(len(unlabeled_examples), dtype=bool)
    classifier_dataloader = generate_data_loader(unlabeled_examples, classifier_label_masks, label_map, batch_size, 1, do_shuffle = False, balance_label_examples = apply_balance)

# Convert the input examples into dataloader

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
  


    transformer.eval() 
    discriminator.eval()
    generator.eval()

    all_preds = []
    all_labels_ids = []
    t0 = time.time()    
    for step,batch in enumerate(classifier_dataloader):
    

        # Unpack this training batch from our dataloader. 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        
            model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]
            _, logits, probs = discriminator(hidden_states)
        ###log_probs = F.log_softmax(probs[:,1:], dim=-1)
            filtered_logits = logits[:,0:-1]
       
            
        # Accumulate the predictions and the input labels
        _, preds = torch.max(filtered_logits, 1)
        all_preds += preds.detach().cpu()
        all_labels_ids += b_labels.detach().cpu()


    
# Report the final accuracy for this validation run.
    all_preds = torch.stack(all_preds).numpy() # list of all the predictions for the data
    all_labels_ids = torch.stack(all_labels_ids).numpy()

    
    return all_preds