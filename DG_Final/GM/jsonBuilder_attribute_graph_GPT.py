import numpy as np
import pandas as pd
import json
import random
import csv
def csv2dict(csvfile):
    '''
    idBefore,item_title,idAfter
    136869,Tinker Bell,0
    3515,Julie &amp; Julia,1
    39456,Santa Buddies: The Legend Of Santa Paws,2
    151398,Mr. Peabody &amp; Sherman,3
    155065,The Hundred-Foot Journey,4
    152638,End of Days,5
    66142,Saving Private Ryan VHS,6
    '''
    dict_list = []
    with open(csvfile, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dict_list.append(row)
    return dict_list
dict_list_M=csv2dict('./item_listM_F.csv')
dict_list_G=csv2dict('./item_listG_AM_F.csv')
json_outp_M=[]
json_outp_G=[]
for item in dict_list_M:
    tmp_dict={}
    tmp_dict["instruction"]="""You will serve as an assistant to help me to extract a list of attributes of the following movies:(For example, when given \
    the movie "The Matrix",you should be able to introduce the movie then extract the attributes ["Action", "Sci-Fi", "Thriller", "Adventure", "Mystery", "Fantasy"]\
    and you should answer with only a list in formact ["",""], with no other words or characters."""
    tmp_dict["input"]="The movie is "+item["item_title"]
    tmp_dict["output"]=""
    json_outp_M.append(tmp_dict)
for item in dict_list_G:
    tmp_dict={}
    tmp_dict["instruction"]="""You will serve as an assistant to help me to extract a list of attributes of the following games or toys:(For example, when given \
    the game or toys Thomas and Friends Wooden Railway,you should be able to introduce the movie then extract the attributs ["Toy", "Thomas & Friends", "Wooden", "Imaginative Play"]\
    and you should answer with only a summary and a list in formact ["",""], with no other words or characters."""
    tmp_dict["input"]="The game is "+item["item_title"]
    tmp_dict["output"]=""
    json_outp_G.append(tmp_dict)
print(dict_list_M[0])
print(json_outp_M[0])
print(dict_list_G[0])
print(json_outp_G[0])
with open('./json_outp_M.json', 'w') as json_file:
    json.dump(json_outp_M, json_file,indent=1)
with open('./json_outp_G.json', 'w') as json_file:
    json.dump(json_outp_G, json_file,indent=4)
    