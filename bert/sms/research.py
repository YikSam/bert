'''from transformers import BertTokenizer, BertForTokenClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)

outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits'''

from transformers import BertTokenizer, BertForTokenClassification
import torch

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
model = BertForTokenClassification.from_pretrained("hfl/chinese-bert-wwm-ext")

sequence = "【人保财险】报案号：RDAA2020420100S0000858，颜廷胜，15040372337，鄂A88888，梅赛德斯-奔驰BJ7204GEL轿车，出险时间：2020-01-013日15:25:58，出险地点：武汉市黄陂区 武湖农场，出险经过：四星  擦到石墩子 本车损 无人伤 现场。请回复：0-不送修，1-送修，2-不确定."

#inputs = tokenizer("【人保财险】报案号：RDAA2020420100S0000858，颜廷胜，15040372337，鄂A88888，梅赛德斯-奔驰BJ7204GEL轿车，出险时间：2020-01-013日15:25:58，出险地点：武汉市黄陂区 武湖农场，出险经过：四星  擦到石墩子 本车损 无人伤 现场。请回复：0-不送修，1-送修，2-不确定.")
#labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1

tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))

print(tokens)
#outputs = model(**inputs, labels=labels, return_dict=True)
#loss = outputs.loss

#logits = outputs.logits

'''sms = ["", ""]
sms_labels = [] 
tokenized = sms.apply((lambda x: tokenizer(x, return_tensors="pt")))
'''




