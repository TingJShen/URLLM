import numpy as np
import random
import scipy.sparse as sp
import torch
import codecs
import json
import copy
from collections import Counter
import torch
import os
import math
import json
from tqdm import tqdm
import numpy as np
import csv
import pandas as pd
import os
import json
import string
import re
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(str(s)))))).strip()

def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        # 在这里可以处理 JSON 数据，例如打印或进行其他操作
        #print(file_path,len(data))
    return data

def traverse_directory(directory_path):
    datas=[]
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith('.json') and file_name.startswith('GM_exat'):
                file_path = os.path.join(root, file_name)
                data=process_json_file(file_path)
                datas+=data
        #print(directory_path,len(datas))
    return datas


class GraphMaker_attribute(object):
    def __init__(self, opt, dirname,id,dirnameItem):
        if id == 'GM':
            self.opt = opt
            self.user = set()
            self.item = set()
            attribute_dict={}
            first_items = []
            data = pd.read_csv('/data/tjshen/Amazon/mg_final/item_listM_F.csv',sep=',',header=0).values[0::,0::]
            for item in data:
                first_items.append([item[1]])
            split_length=len(first_items)
            data = pd.read_csv('/data/tjshen/Amazon/mg_final/item_listG_AM_F.csv',sep=',',header=0).values[0::,0::]
            for item in data:
                first_items.append([item[1]])
            first_items.append([''])     #item-user-attribute item
            print("len(first_items)",len(first_items))
            user_dict={}
            directory_to_traverse = dirname
            entrys=traverse_directory(directory_to_traverse)
            item_GM_dict={}
            directory_to_traverse_GM=dirnameItem
            entrys_GM=traverse_directory(directory_to_traverse_GM)
            for entry in tqdm(entrys_GM):
                item_GM_dict[entry['qqid']]=list(set(normalize_answer(entry['choices'][0]['message']['content']).split()))
            item_GM_dict[len(first_items)-1]=''
            #attribute2user (userid+len(first_items))
            for entry in tqdm(entrys):
                user_dict[entry['qqid']+len(first_items)]=list(set(normalize_answer(entry['choices'][0]['message']['content']).split()))
                for attribute in user_dict[entry['qqid']+len(first_items)]:
                    if attribute not in attribute_dict:
                        attribute_dict[attribute]=[len(attribute_dict),1,0,0]
                    else:
                        attribute_dict[attribute][1]+=1
            for user_check_idx in range(opt['usernum']):
                if (user_check_idx+len(first_items)) not in user_dict:
                    user_dict[user_check_idx+len(first_items)]=['Game','Movie']
            print('#2',len(user_dict),len(attribute_dict))
            item_dict={}
            #attribute2user (userid+len(first_items))
            for idx in tqdm(range(len(first_items))):
                item_dict[idx]=item_GM_dict[idx]
                for item in item_dict[idx]:
                    if item not in attribute_dict:
                        if idx < split_length:
                            #atrribute in source domain
                            attribute_dict[item]=[len(attribute_dict),0,1,0]
                        else:
                            #atrribute in target domain
                            attribute_dict[item]=[len(attribute_dict),0,0,1]
                    else:
                        if idx < split_length:
                            #atrribute in source domain
                            attribute_dict[item][2]+=1
                        else:
                            #atrribute in target domain
                            attribute_dict[item][3]+=1
            print('#3',len(item_dict),len(attribute_dict))
                    
            cnt=0
            cnt_2_1=0
            cnt_2_2=0
            for attribute in attribute_dict:
                if (opt['GAC']==1 or attribute_dict[attribute][1]>1) and (attribute_dict[attribute][2]>1 or attribute_dict[attribute][3]>1):
                    cnt_2_1+=attribute_dict[attribute][1]
                    cnt_2_2+=attribute_dict[attribute][2]
                    cnt+=1
            print('#4 Gcnt',cnt)
            print('#5 Gcnt',cnt_2_1,cnt_2_2,cnt_2_1+cnt_2_2)

            attribute_rerank_dict={}
            attribute_rerank_dict_source={}
            attribute_rerank_dict_target={}
            for attribute in attribute_dict:
                if (opt['GAC']==1 or attribute_dict[attribute][1]>1) and (attribute_dict[attribute][2]>1 or attribute_dict[attribute][3]>1):
                    attribute_rerank_dict[attribute]=len(attribute_rerank_dict)+len(first_items)+len(user_dict)
                    # attribute区分source与target
                    if attribute_dict[attribute][2]>1:
                        attribute_rerank_dict_source[attribute]=len(attribute_rerank_dict)+len(first_items)+len(user_dict)# item-user-attribute
                    if attribute_dict[attribute][3]>1:
                        attribute_rerank_dict_target[attribute]=len(attribute_rerank_dict)+len(first_items)+len(user_dict)

            graph=[]
            graph_source=[]
            graph_target=[]
            '''
            for user in tqdm(user_dict):
                for attribute in user_dict[user]:
                    if attribute in attribute_rerank_dict:
                        #undirected graph
                        graph.append([user,attribute_rerank_dict[attribute]])
                        graph.append([attribute_rerank_dict[attribute],user])
            '''
            for item in tqdm(item_dict):
                for attribute in item_dict[item]:
                    if attribute in attribute_rerank_dict:
                        graph.append([item,attribute_rerank_dict[attribute]])
                        graph.append([attribute_rerank_dict[attribute],item])
                        if item < split_length and attribute in attribute_rerank_dict_source:
                            graph_source.append([attribute_rerank_dict[attribute],item])
                        if item >= split_length and attribute in attribute_rerank_dict_target:
                            graph_target.append([item,attribute_rerank_dict[attribute]])
            print('#6',len(graph),len(graph_source),len(graph_target))
            print('#7',len(attribute_rerank_dict),len(attribute_rerank_dict_source),len(attribute_rerank_dict_target))
            print('#8,',len(user_dict),len(item_dict),len(attribute_rerank_dict),len(user_dict)+len(item_dict)+len(attribute_rerank_dict))
            graph=np.array(graph)
            graph_source=np.array(graph_source)
            graph_target=np.array(graph_target)
            adj=sp.coo_matrix((np.ones(graph.shape[0]), (graph[:, 0], graph[:, 1])),
                                shape=(len(user_dict)+len(item_dict)+len(attribute_rerank_dict), len(user_dict)+len(item_dict)+len(attribute_rerank_dict)),
                                dtype=np.float32)
            adj_source=sp.coo_matrix((np.ones(graph_source.shape[0]), (graph_source[:, 0], graph_source[:, 1])),
                                shape=(len(user_dict)+len(item_dict)+len(attribute_rerank_dict), len(user_dict)+len(item_dict)+len(attribute_rerank_dict)),
                                dtype=np.float32)
            adj_target=sp.coo_matrix((np.ones(graph_target.shape[0]), (graph_target[:, 0], graph_target[:, 1])),
                                shape=(len(user_dict)+len(item_dict)+len(attribute_rerank_dict), len(user_dict)+len(item_dict)+len(attribute_rerank_dict)),
                                dtype=np.float32)
            adj = normalize(adj)
            adj_source = normalize(adj_source)
            adj_target = normalize(adj_target)
            adj = sparse_mx_to_torch_sparse_tensor(adj)
            adj_source = sparse_mx_to_torch_sparse_tensor(adj_source)
            adj_target = sparse_mx_to_torch_sparse_tensor(adj_target)
            self.adj_att = adj
            self.adj_att_source = adj_source
            self.adj_att_target = adj_target
            self.attribute_len=len(attribute_rerank_dict)
        if id.startswith('AO_J'):
            self.opt = opt
            self.user = set()
            self.item = set()
            attribute_dict={}
            first_items = []
            data = pd.read_csv('../AO/item_listA_F.csv',sep=',',header=0).values[0::,0::]
            for item in data:
                first_items.append([item[1],item[3]])
            split_length=len(first_items)
            data = pd.read_csv('../AO/item_listO_AA_F.csv',sep=',',header=0).values[0::,0::]
            for item in data:
                first_items.append([item[1],item[3]])
            first_items.append(['',''])     #item-user-attribute item
            item_dict={}
            #attribute2user (userid+len(first_items))
            for idx in tqdm(range(len(first_items))):
                item_dict[idx]=list(set(normalize_answer(first_items[idx][1]).split()))
            if id == 'AO_J':
                print('loading AO_J')
                source_json=json.load(open('./dataset/graph_AO_jqz/graph_source.json','r'))
                target_json=json.load(open('./dataset/graph_AO_jqz/graph_target.json','r'))
                full_json=json.load(open('./dataset/graph_AO_jqz/graph.json','r'))
            if id == 'AO_JI':
                print('loading AO_JI')
                source_json=json.load(open('./dataset/graph_i_aug_AO/graph_source.json','r'))
                target_json=json.load(open('./dataset/graph_i_aug_AO/graph_target.json','r'))
                full_json=json.load(open('./dataset/graph_i_aug_AO/graph.json','r'))
            if id == 'AO_JA':
                print('loading AO_JA')
                source_json=json.load(open('./dataset/graph_all_aug_AO/graph_source.json','r'))
                target_json=json.load(open('./dataset/graph_all_aug_AO/graph_target.json','r'))
                full_json=json.load(open('./dataset/graph_all_aug_AO/graph.json','r'))
            if id == 'AO_JAC':
                print('loading AO_JAC')
                source_json=json.load(open('./dataset/graph_all_aug_AO_C/graph_source.json','r'))
                target_json=json.load(open('./dataset/graph_all_aug_AO_C/graph_target.json','r'))
                full_json=json.load(open('./dataset/graph_all_aug_AO_C/graph.json','r'))
            if id == 'AO_JIC':
                print('loading AO_JIC')
                source_json=json.load(open('./dataset/graph_i_aug_AO_C/graph_source.json','r'))
                target_json=json.load(open('./dataset/graph_i_aug_AO_C/graph_target.json','r'))
                full_json=json.load(open('./dataset/graph_i_aug_AO_C/graph.json','r'))
            if id == 'AO_JIC_fonly':
                print('loading AO_JIC_fonly')
                source_json=json.load(open('./dataset/graph_i_aug_AO_IC_fonly/graph_source.json','r'))
                target_json=json.load(open('./dataset/graph_i_aug_AO_IC_fonly/graph_target.json','r'))
                full_json=json.load(open('./dataset/graph_i_aug_AO_IC_fonly/graph.json','r'))
            full_json_num=[]
            source_json_num=[]
            target_json_num=[]
            for item in full_json:
                full_json_num.append([int(item[0]),int(item[1])])
            for item in source_json:
                source_json_num.append([int(item[0]),int(item[1])])
            for item in target_json:
                target_json_num.append([int(item[0]),int(item[1])])
            graph=np.array(full_json_num)
            graph_source=np.array(source_json_num)
            graph_target=np.array(target_json_num)
            print('##graph',graph[0:10])
            max_idx=-999
            for item in graph:
                for idx_item in item:
                    if idx_item >=max_idx:
                        max_idx=idx_item
            max_idx+=1
            print('##max_idx',max_idx)
            adj=sp.coo_matrix((np.ones(graph.shape[0]), (graph[:, 0], graph[:, 1])),
                                shape=(max_idx, max_idx),
                                dtype=np.float32)
            adj_source=sp.coo_matrix((np.ones(graph_source.shape[0]), (graph_source[:, 0], graph_source[:, 1])),
                                shape=(max_idx, max_idx),
                                dtype=np.float32)
            adj_target=sp.coo_matrix((np.ones(graph_target.shape[0]), (graph_target[:, 0], graph_target[:, 1])),
                                shape=(max_idx, max_idx),
                                dtype=np.float32)
            adj = normalize(adj)
            adj_source = normalize(adj_source)
            adj_target = normalize(adj_target)
            adj = sparse_mx_to_torch_sparse_tensor(adj)
            adj_source = sparse_mx_to_torch_sparse_tensor(adj_source)
            adj_target = sparse_mx_to_torch_sparse_tensor(adj_target)
            self.adj_att = adj
            self.adj_att_source = adj_source
            self.adj_att_target = adj_target
            self.attribute_len=max_idx-len(item_dict)-19154
        if id == 'AO':
            self.opt = opt
            self.user = set()
            self.item = set()
            attribute_dict={}
            first_items = []
            data = pd.read_csv('/data/tjshen/ART_OFFICE_CDR_FINAL/last/item_listA_F.csv',sep=',',header=0).values[0::,0::]
            for item in data:
                first_items.append([item[1],item[3]])
            split_length=len(first_items)
            data = pd.read_csv('/data/tjshen/ART_OFFICE_CDR_FINAL/last/item_listO_AA_F.csv',sep=',',header=0).values[0::,0::]
            for item in data:
                first_items.append([item[1],item[3]])
            first_items.append(['',''])     #item-user-attribute item
            #print("len(first_items)",len(first_items))
            user_dict={}
            directory_to_traverse = dirname
            entrys=traverse_directory(directory_to_traverse)
            #attribute2user (userid+len(first_items))
            for entry in tqdm(entrys):
                user_dict[entry['qqid']+len(first_items)]=list(set(normalize_answer(entry['choices'][0]['message']['content']).split()))
                for attribute in user_dict[entry['qqid']+len(first_items)]:
                    if attribute not in attribute_dict:
                        attribute_dict[attribute]=[len(attribute_dict),1,0,0]
                    else:
                        attribute_dict[attribute][1]+=1
            #print('#2',len(user_dict),len(attribute_dict))
            item_dict={}
            #attribute2user (userid+len(first_items))
            for idx in tqdm(range(len(first_items))):
                item_dict[idx]=list(set(normalize_answer(first_items[idx][1]).split()))
                for item in item_dict[idx]:
                    if item not in attribute_dict:
                        if idx < split_length:
                            #atrribute in source domain
                            attribute_dict[item]=[len(attribute_dict),0,1,0]
                        else:
                            #atrribute in target domain
                            attribute_dict[item]=[len(attribute_dict),0,0,1]
                    else:
                        if idx < split_length:
                            #atrribute in source domain
                            attribute_dict[item][2]+=1
                        else:
                            #atrribute in target domain
                            attribute_dict[item][3]+=1
            #print('#3',len(item_dict),len(attribute_dict))
                    
            cnt=0
            cnt_2_1=0
            cnt_2_2=0
            for attribute in attribute_dict:
                if (opt['GAC']==1 or attribute_dict[attribute][1]>1) and (attribute_dict[attribute][2]>1 or attribute_dict[attribute][3]>1):
                    cnt_2_1+=attribute_dict[attribute][1]
                    cnt_2_2+=attribute_dict[attribute][2]
                    cnt+=1
            print('#4 Gcnt',cnt)
            print('#5 Gcnt',cnt_2_1,cnt_2_2,cnt_2_1+cnt_2_2)

            attribute_rerank_dict={}
            attribute_rerank_dict_source={}
            attribute_rerank_dict_target={}
            for attribute in attribute_dict:
                if (opt['GAC']==1 or attribute_dict[attribute][1]>1) and (attribute_dict[attribute][2]>1 or attribute_dict[attribute][3]>1):
                    attribute_rerank_dict[attribute]=len(attribute_rerank_dict)+len(first_items)+len(user_dict)
                    # attribute区分source与target
                    if attribute_dict[attribute][2]>1:
                        attribute_rerank_dict_source[attribute]=len(attribute_rerank_dict)+len(first_items)+len(user_dict)# item-user-attribute
                    if attribute_dict[attribute][3]>1:
                        attribute_rerank_dict_target[attribute]=len(attribute_rerank_dict)+len(first_items)+len(user_dict)

            graph=[]
            graph_source=[]
            graph_target=[]
            '''
            for user in tqdm(user_dict):
                for attribute in user_dict[user]:
                    if attribute in attribute_rerank_dict:
                        #undirected graph
                        graph.append([user,attribute_rerank_dict[attribute]])
                        graph.append([attribute_rerank_dict[attribute],user])
            '''
            for item in tqdm(item_dict):
                for attribute in item_dict[item]:
                    if attribute in attribute_rerank_dict:
                        graph.append([item,attribute_rerank_dict[attribute]])
                        graph.append([attribute_rerank_dict[attribute],item])
                        if item < split_length and attribute in attribute_rerank_dict_source:
                            graph_source.append([attribute_rerank_dict[attribute],item])
                        if item >= split_length and attribute in attribute_rerank_dict_target:
                            graph_target.append([item,attribute_rerank_dict[attribute]])
            #print('#6',len(graph),len(graph_source),len(graph_target))
            graph=np.array(graph)
            graph_source=np.array(graph_source)
            graph_target=np.array(graph_target)
            adj=sp.coo_matrix((np.ones(graph.shape[0]), (graph[:, 0], graph[:, 1])),
                                shape=(len(user_dict)+len(item_dict)+len(attribute_rerank_dict), len(user_dict)+len(item_dict)+len(attribute_rerank_dict)),
                                dtype=np.float32)
            adj_source=sp.coo_matrix((np.ones(graph_source.shape[0]), (graph_source[:, 0], graph_source[:, 1])),
                                shape=(len(user_dict)+len(item_dict)+len(attribute_rerank_dict), len(user_dict)+len(item_dict)+len(attribute_rerank_dict)),
                                dtype=np.float32)
            adj_target=sp.coo_matrix((np.ones(graph_target.shape[0]), (graph_target[:, 0], graph_target[:, 1])),
                                shape=(len(user_dict)+len(item_dict)+len(attribute_rerank_dict), len(user_dict)+len(item_dict)+len(attribute_rerank_dict)),
                                dtype=np.float32)
            adj = normalize(adj)
            adj_source = normalize(adj_source)
            adj_target = normalize(adj_target)
            adj = sparse_mx_to_torch_sparse_tensor(adj)
            adj_source = sparse_mx_to_torch_sparse_tensor(adj_source)
            adj_target = sparse_mx_to_torch_sparse_tensor(adj_target)
            self.adj_att = adj
            self.adj_att_source = adj_source
            self.adj_att_target = adj_target
            self.attribute_len=len(attribute_rerank_dict)
            #input()



