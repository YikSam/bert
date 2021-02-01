from transformers import BertForTokenClassification, BertTokenizer
import torch

model = BertForTokenClassification.from_pretrained("C:\\ext\\codes\\bert\\bert\\sms\\output")
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm", do_lower_case=True)

'''label_list = [
...     "O",       # Outside of a named entity
...     "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
...     "I-MISC",  # Miscellaneous entity
...     "B-PER",   # Beginning of a person's name right after another person's name
...     "I-PER",   # Person's name
...     "B-ORG",   # Beginning of an organisation right after another organisation
...     "I-ORG",   # Organisation
...     "B-LOC",   # Beginning of a location right after another location
...     "I-LOC"    # Location
... ]'''

label_list = ['CUST', 'C_NO', 'MOBILE', 'LOC', 'PLATE', 'VTYPE', 'CMPY', 'O']

sequence = "【太平洋保险】太平洋保险提醒您，车牌号桂NSC889奔驰BENZ GLE320越野车车险已报案，出险时间:2021年01月14日18时48分04秒，出险地点：广西壮族自治区南宁市青秀区双拥路辅路，请跟进（奔驰基本2020）钣喷维修专享礼服务，车主姓名：甘圣海,车险报案电话:18077706009。谢谢。 回TD退订短信"
#sequence = "【太平洋保险】太平洋保险提醒您，车牌号桂A3855M梅赛德斯-奔驰BJ7167J轿车车险已报案，出险时间:2021年01月14日18时00分13秒，出险地点：南宁市白沙大道冠星奔驰，请跟进（奔驰基本2020）钣喷维修专享礼服务，车主姓名：陆文景,车险报案电话:13877149957。谢谢。 回TD退订短信"
#sequence = "【太平洋产险】武汉星隆,我司已推荐三者车颜廷胜15040372337的鄂AEF888送至您厂维修。 回TD退订短"

sequence = " ".join([s[0] for s in sequence])

tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
inputs = tokenizer.encode(sequence, return_tensors="pt")

outputs = model(inputs)[0]
predictions = torch.argmax(outputs, dim=2)

print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())])