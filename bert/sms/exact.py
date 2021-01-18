import numpy as np
import torch
import pandas as pd

df = pd.read_csv('C:\\ext\\codes\\bert\\bert\\sms\\accident_lead_info.csv').query('dmp_success == 1')
df1 = pd.DataFrame(columns=['Sentence #', 'Word', 'Tag'])

map = {
    1: 'CMPY',
    2: 'PLATE',
    3: 'C_NO',
    4: 'CUST',
    5: 'MOBILE',
    6: 'VTYPE',
    7: 'LOC'
}

for index, row in df.iterrows():

    text = row['original_sms']
    print(text)

    words = list(text)
    tags = np.zeros(len(words))

    if not row['insurance_company'] is np.nan:
        start = text.find(row['insurance_company'])
        if start > -1:
            for i in range(start, start + len(row['insurance_company'])):
                tags[i] = 1

    if not row['plate'] is np.nan:
        start = text.find(row['plate'])
        if start > -1:
            for i in range(start, start + len(row['plate'])):
                tags[i] = 2
    
    if not row['insurance_case_number'] is np.nan:
        start = text.find(row['insurance_case_number'])
        if start > -1:
            for i in range(start, start + len(row['insurance_case_number'])):
                tags[i] = 3
    
    if not row['customer_name'] is np.nan:
        start = text.find(row['customer_name'])
        if start > -1:
            for i in range(start, start + len(row['customer_name'])):
                tags[i] = 4

    if not row['mobile'] is np.nan:
        start = text.find(row['mobile'])
        if start > -1:
            for i in range(start, start + len(row['mobile'])):
                tags[i] = 5

    if not row['vehicle_type'] is np.nan:
        start = text.find(row['vehicle_type'])
        if start > -1:
            for i in range(start, start + len(row['vehicle_type'])):
                tags[i] = 6

    if not row['vehicle_location'] is np.nan:
        start = text.find(row['vehicle_location'])
        if start > -1:
            for i in range(start, start + len(row['vehicle_location'])):
                tags[i] = 7

    
    for i in range(len(list) - 1):
        name = ""
        if i == 0:
            name = "Sentence: {}".format(index + 1)

        tag = map[tags[i]]
        
        df1.append(pd.DataFrame({
            'Sentence #': name,
            'Word': list[i],
            'Tag': tag}))

    


'''text = "【人保财险】报案号：RDAA2020420100S0000858，颜廷胜，15040372337，鄂A88888，梅赛德斯-奔驰BJ7204GEL轿车，出险时间：2020-01-013日15:25:58，出险地点：武汉市黄陂区 武湖农场，出险经过：四星  擦到石墩子 本车损 无人伤 现场。请回复：0-不送修，1-送修，2-不确定."

#list = ['[CLS]', '【', '人', '保', '财', '险', '】', '报', '案', '号', '：', 'rd', '##aa', '##20', '##20', '##42', '##010', '##0', '##s', '##000', '##08', '##58', '，', '颜', '廷', '胜', '，', '150', '##40', '##37', '##23', '##37', '，', '鄂', 'a8', '##888', '##8', '，', '梅', '赛', '德', '斯', '-', '奔', '驰', 'b', '##j', '##72', '##04', '##ge', '##l', '轿', '车', '，', '出', '险', '时', '间', '：', '2020', '-', '01', '-', '013', '日', '15', ':', '25', ':', '58', '，', '出', '险', '地', '点', '：', '武', '汉', '市', '黄', '陂', '区', '武', '湖', '农', '场', '，', '出', '险', '经', '过', '：', '四', '星', '擦', '到', '石', '墩', '子', '本', '车', '损', '无', '人', '伤', '现', '场', '。', '请', '回', '复', '：', '0', '-', '不', '送', '修', '，', '1', '-', '送', '修', '，', '2', '-', '不', '确', '定', '.', '[SEP]']

token_cmpy = '人保财险'
index_cmpy = 0
label_cmpy = []

token_plate = '鄂A88888'.lower()
index_plate = 0
label_plate = []

token_cs_no = 'RDAA2020420100S0000858'.lower()
index_cs_no = 0
label_cs_no = []

token_cust_nm = '颜廷胜'.lower()
index_cust_nm = 0
label_cust_nm = []

token_mobile = '15040372337'.lower()
index_mobile = 0
label_mobile = []

token_vtype = '梅赛德斯-奔驰BJ7204GEL轿车'.lower()
index_vtype = 0
label_vtype = []

token_loc = '武汉市黄陂区 武湖农场'.replace(' ', '').lower()
index_loc = 0
label_loc = []


#label = torch.zeros(len(list))

i = 0

while i < len(text):
    print(text[i])
    if list[i].replace('##', '') in token_cmpy:
        label_cmpy_temp = []
        test_cmpy_temp = ""
        while len(test_cmpy_temp) < len(token_cmpy):
            test_cmpy_temp += list[i + index_cmpy].replace('##', '')
            if test_cmpy_temp not in token_cmpy:
                break
            label_cmpy_temp.append(i + index_cmpy)
            index_cmpy += 1
        if token_cmpy == test_cmpy_temp:
            label_cmpy = label_cmpy_temp.copy()
            i += index_cmpy - 1
        else:
            index_cmpy = 0
    
    if list[i].replace('##', '') in token_plate:
        label_plate_temp = []
        test_plate_temp = ""
        while len(test_plate_temp) < len(token_plate):
            test_plate_temp += list[i + index_plate].replace('##', '')
            if test_plate_temp not in token_plate:
                break
            label_plate_temp.append(i + index_plate)
            index_plate += 1
        if token_plate == test_plate_temp:
            label_plate = label_plate_temp.copy()
            i += index_plate - 1
        else:
            index_plate = 0
    
    if list[i].replace('##', '') in token_cs_no:
        label_cs_no_temp = []
        test_cs_no_temp = ""
        while len(test_cs_no_temp) < len(token_cs_no):
            test_cs_no_temp += list[i + index_cs_no].replace('##', '')
            if test_cs_no_temp not in token_cs_no:
                break
            label_cs_no_temp.append(i + index_cs_no)
            index_cs_no += 1
        if token_cs_no == test_cs_no_temp:
            label_cs_no = label_cs_no_temp.copy()
            i += index_cs_no - 1
        else:
            index_cs_no = 0
    
    if list[i].replace('##', '') in token_cust_nm:
        label_cust_nm_temp = []
        test_cust_nm_temp = ""
        while len(test_cust_nm_temp) < len(token_cust_nm):
            test_cust_nm_temp += list[i + index_cust_nm].replace('##', '')
            if test_cust_nm_temp not in token_cust_nm:
                break
            label_cust_nm_temp.append(i + index_cust_nm)
            index_cust_nm += 1
        if token_cust_nm == test_cust_nm_temp:
            label_cust_nm = label_cust_nm_temp.copy()
            i += index_cust_nm - 1
        else:
            index_cust_nm = 0

    if list[i].replace('##', '') in token_mobile:
        label_mobile_temp = []
        test_mobile_temp = ""
        while len(test_mobile_temp) < len(token_mobile):
            test_mobile_temp += list[i + index_mobile].replace('##', '')
            if test_mobile_temp not in token_mobile:
                break
            label_mobile_temp.append(i + index_mobile)
            index_mobile += 1
        if token_mobile == test_mobile_temp:
            label_mobile = label_mobile_temp.copy()
            i += index_mobile - 1
        else:
            index_mobile = 0

    if list[i].replace('##', '') in token_vtype:
        label_vtype_temp = []
        test_vtype_temp = ""
        while len(test_vtype_temp) < len(token_vtype):
            test_vtype_temp += list[i + index_vtype].replace('##', '')
            if test_vtype_temp not in token_vtype:
                break
            label_vtype_temp.append(i + index_vtype)
            index_vtype += 1
        if token_vtype == test_vtype_temp:
            label_vtype = label_vtype_temp.copy()
            i += index_vtype - 1
        else:
            index_vtype = 0
    
    if list[i].replace('##', '') in token_loc:
        label_loc_temp = []
        test_loc_temp = ""
        while len(test_loc_temp) < len(token_loc):
            test_loc_temp += list[i + index_loc].replace('##', '')
            if test_loc_temp not in token_loc:
                break
            label_loc_temp.append(i + index_loc)
            index_loc += 1
        if token_loc == test_loc_temp:
            label_loc = label_loc_temp.copy()
            i += index_loc - 1
        else:
            index_loc = 0

    i += 1 '''

print(label_cmpy)