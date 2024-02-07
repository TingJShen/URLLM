import numpy as np
import pandas as pd
import json
import random
from tqdm import tqdm
data = pd.read_csv('./item_listA_F.csv',sep=',',header=0).values[0::,0::]
first_items=[]
for item in data:
    first_items.append(item[1])
split_len=len(first_items)
print(first_items[0],len(first_items))
data = pd.read_csv('./item_listO_AA_F.csv',sep=',',header=0).values[0::,0::]
#first_items=[]
for item in data:
    first_items.append(item[1])
    
print(first_items[split_len],len(first_items))

sample_num = 20

similar_seq=[] # is a[][]
case=open('./train_F.txt','r')
lines = case.readlines()
for line in lines:
    record=line.split('\t')
    tmp_list=[]
    for item_idx in range(2,len(record)-1,1):
        item=int(record[item_idx].split('|')[0].replace('\n',''))
        tmp_list.append(item)
    similar_seq.append(tmp_list)

def calculate_metrics(predicted, actual):
    true_positives = len(set(predicted) & set(actual))
    if true_positives == 0:
        return 0, 0, 0

    false_positives = len(set(predicted) - set(actual))
    false_negatives = len(set(actual) - set(predicted))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


output_json=[]
max_tk=512
case=open('./test_F.txt','r')
lines = case.readlines()
case.close()
matmul_trte=np.load('../DG_Final/DG_src/t4_G2_final_DGresult_matmul_trte.npy') #[3601*35900]
#for line_idx in range(2,3,1):
HR_list=[]
len_list=[]
ground_truths=[]
for line_idx in tqdm(range(len(lines))):
    line=lines[line_idx]
    tmp_dict={}
    tmp_dict["instruction"]="Given a list of arts or office the user has bought before, please recommend a new arts or office that the user likes to the user."
    tmp_dict["input"]="The user has played the following arts or office before:"
    tmp_line=line.split('\t') 
    tmp_end_word=""
    tmp_seq_list=[]
    for case in range(2,len(tmp_line)-1,1):
        case_id=int(tmp_line[case].split('|')[0].replace('\n',''))
        if((tmp_line[case].split('|')[1].replace('\n',''))!=(tmp_line[1])):
            tmp_seq_list.append(case_id)
            if case_id>=split_len:
                tmp_dict["input"]=tmp_dict["input"]+" office:"+first_items[case_id]+' |'
            else:
                tmp_dict["input"]=tmp_dict["input"]+" art:"+first_items[case_id]+' |'

        else:
            ground_truth=int(tmp_line[case].split('|')[0].replace('\n',''))
            ground_truths.append(ground_truth)
            if int(tmp_line[case].split('|')[0].replace('\n',''))<split_len: 
                tmp_end_word='Please recommend a new art\n'
            else:
                tmp_end_word='Please recommend a new office\n'
            tmp_dict["output"]=first_items[int(tmp_line[case].split('|')[0].replace('\n',''))]
    argsort_trte=np.argsort(matmul_trte[line_idx])
    tmp_input=''
    full_seq=[]
    for ii in range(0,1,1):
        tmp_input+='There is another similar user who has played arts or office before:'
        full_seq+=similar_seq[argsort_trte[ii]]
        for item_train in similar_seq[argsort_trte[ii]]: 
            if ((len(tmp_dict["input"].split(' '))+len(tmp_input.split()))<max_tk):
                if item_train < split_len:
                    tmp_input=tmp_input+" art:"+str(first_items[item_train])+' |'
                else:
                    tmp_input=tmp_input+" office:"+str(first_items[item_train])+' |'
    len_list.append(len(full_seq))
    if ground_truth in full_seq:
        HR_list.append(1)
    else:
        HR_list.append(0)
    #    print('##seq_max',seq_max[ii],'##f1',f1_scores[seq_max[ii]])
    #    print(similar_seq[seq_max[ii]])



    tmp_dict["input"]=tmp_input+tmp_dict["input"]+tmp_end_word
    output_json.append(tmp_dict)
    #print(int(tmp_line[0]),line[0:40])


def DCG(A, test_set,k):
    # ------ 计算 DCG ------ #
    dcg = 0
    for i in range(k):
        # 给r_i赋值，若r_i在测试集中则为1，否则为0
        r_i = 0
        if A[i] == test_set:
            r_i = 1
        dcg += (2 ** r_i - 1) / np.log2((i + 1) + 1) # (i+1)是因为下标从0开始
    return dcg

def IDCG(A, test_set,k):
    return 1 #all at Top 1

def NDCG(test_set,A,k):
    ndcg_avg=0
    for i in range(len(A)):
        ndcg_avg+=DCG(A[i],test_set[i],k)
    return ndcg_avg/len(A)

def Recall(test_set,A,k):
    recall_avg=0
    for i in range(len(A)):
        if test_set[i] in A[i][0:k]:
            recall_avg+=1
    return recall_avg/len(A)
ground_truths=np.array(ground_truths)

HR=np.mean(HR_list)
mean_len=np.mean(len_list)
print(mean_len)
print(HR)


output_json_ = json.dumps(output_json, ensure_ascii=False,indent=1)
with open("./test_similar_users.json", "w") as file:
    file.write(output_json_)


