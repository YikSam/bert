from transformers import DistilBertTokenizer, DistilBertModel

import torch
from torch.nn import Linear
from torch.nn import Dropout
from torch.nn import CrossEntropyLoss

from datasets import load_dataset

#tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#model = DistilBertModel.from_pretrained('distilbert-base-uncased')

datasets = load_dataset('squad', split='train')


print(datasets)

#qa_outputs = Linear(768, 2)
#qa_droupout = Dropout(qa_outputs)



#inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
#start_positions = torch.tensor([1])
#end_positions = torch.tensor([3])

#outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
#loss = outputs.loss

#results_start = torch.softmax(outputs.start_logits, dim=1)
#result_end = torch.softmax(outputs.end_logits, dim=1)

#print(results_start)