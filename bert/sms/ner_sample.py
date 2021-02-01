import numpy as np
import pandas as pd

import os
print(os.listdir('C:\\ext\\codes\\bert\\bert\\input'))

from tqdm import tqdm, trange

#input_data = pd.read_csv("C:\\ext\\codes\\bert\\bert\\input\\ner_dataset.csv", encoding='latin1')
input_data = pd.read_csv("C:\\ext\\codes\\bert\\bert\\sms\\tag_sample.csv", encoding='utf_8', header=0)
input_data = input_data.fillna(method="ffill")
print(input_data.tail(10))

words_list = list(set(input_data["Word"].values))
print(words_list[:10])

number_words = len(words_list); number_words

class RetrieveSentence(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        function = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(function)
        self.sentences = [s for s in self.grouped]
        self.sentences = self.sentences[1:]

    def retrieve(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

Sentences = RetrieveSentence(input_data)
print(Sentences)

Sentences_list = [" ".join([s[0] for s in sent]) for sent in Sentences.sentences]
print(Sentences_list[0])
print(len(Sentences_list))

labels = [[s[1] for s in sent] for sent in Sentences.sentences]
print(labels[0])

tags2vals = [t for t in set(input_data["Tag"].values) if t != 'Tag']
tag2idx = {t: i for i, t in enumerate(tags2vals)}

print(tags2vals)
print(tag2idx)

import torch 
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.utils.rnn as rnn_utils
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import BertForTokenClassification

max_seq_len = 400    # tokens
batch_s = 8        # batch size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(torch.cuda.get_device_name(0))

tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in Sentences_list]
print(tokenized_texts[0])
print(len(tokenized_texts))
print(tokenized_texts[1])

#X = rnn_utils.pad_sequence([torch.Tensor(tokenizer.convert_tokens_to_ids(txt)) for txt in tokenized_texts], batch_first=True)
X = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts], maxlen=max_seq_len, dtype="long", truncating="post", padding="post")
#Y = rnn_utils.pad_sequence([torch.Tensor([tag2idx.get(1) for l in lab]) for lab in labels], batch_first=True)
Y = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels], maxlen=max_seq_len, value=tag2idx["O"], padding="post", dtype="long", truncating="post")
print(X.shape)
print(Y.shape)

attention_masks = [[float(i > 0) for i in ii] for ii in X]
print(len(attention_masks))
print(attention_masks[0])

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, random_state=20, test_size=0.1)
Mask_train, Mask_valid, _, _ = train_test_split(attention_masks, X, random_state=20, test_size=0.1)

X_train = torch.tensor(X_train).to(device).long()
X_valid = torch.tensor(X_valid).to(device).long()
Y_train = torch.tensor(Y_train).to(device).long()
Y_valid = torch.tensor(Y_valid).to(device).long()
Mask_train = torch.tensor(Mask_train).to(device).long()
Mask_valid = torch.tensor(Mask_valid).to(device).long()

data_train = TensorDataset(X_train, Mask_train, Y_train)
data_train_sampler = RandomSampler(data_train)
DL_train = DataLoader(data_train, sampler=data_train_sampler, batch_size=batch_s)

data_valid = TensorDataset(X_valid, Mask_valid, Y_valid)
data_valid_sampler = SequentialSampler(data_valid)
DL_valid = DataLoader(data_valid, sampler=data_valid_sampler, batch_size=batch_s)

model = BertForTokenClassification.from_pretrained("hfl/chinese-bert-wwm", num_labels=len(tag2idx))
model.cuda()

FULL_FINETUNING = False
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

from seqeval.metrics import f1_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

epochs = 50
max_grad_norm = 1.0

for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(DL_train):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss, _ = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in DL_valid:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            tmp_eval_loss, _ = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
        break
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tags2vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags2vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score([pred_tags], [valid_tags])))

model.save_pretrained('C:\\ext\\codes\\bert\\bert\\sms\\output')

model.eval()
predictions = []
true_labels = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in DL_valid:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        tmp_eval_loss, _ = model(b_input_ids, token_type_ids=None,
                              attention_mask=b_input_mask, labels=b_labels)
        logits = model(b_input_ids, token_type_ids=None,
                       attention_mask=b_input_mask)[0]
        
    logits = logits.detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    label_ids = b_labels.to('cpu').numpy()
    true_labels.append(label_ids)
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy

    nb_eval_examples += b_input_ids.size(0)
    nb_eval_steps += 1

pred_tags = [[tags2vals[p_i] for p_i in p] for p in predictions]
valid_tags = [[tags2vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l ]
print("Validation loss: {}".format(eval_loss/nb_eval_steps))
print("Validation accuracy: {}".format(eval_accuracy/nb_eval_steps))
print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

print("finished")