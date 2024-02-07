

import openai
import os
import json
import requests
import numpy as np
import pandas as pd
import json
import random
import argparse
from tqdm import tqdm
import time
def openai_reply(message,apikey,api_url,headers):
    openai.api_key = apikey
    data={
        "model":"gpt-3.5-turbo-0301",
        "messages":message}
    response = requests.post(api_url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        result = response.json()
        return (result)
    else:
        return (f"Error: {response.status_code} - {response.text}")
    '''
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",#gpt-3.5-turbo-0301
    messages=message
    temperature=0.5,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    )
    # print(response)
    return response.choices[0].message.content
    '''





def solve(instrution,input_):
    
    openai.api_key = #your api key
    openai.api_base="https://api.chatanywhere.com.cn"


    
    #print(response)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": instrution},
                {"role": "user", "content": input_}
            ],
            temperature=0.7
        )
        dct = response
        time.sleep(1)
        #print(dct)
        if 'id' in dct:
            #print('success')
            return dct
        else:
            #print('fail')
            return None
    except Exception as e:
        print(e)
        return None

def infer(cluster_n,ii_idx):
    path='./'
    case_M=json.load(open(path+'/json_outp_M.json','r'))
    case_R=json.load(open(path+'/json_outp_G.json','r'))
    output_json=case_M+case_R

    print(len(output_json))
    responses_final=[]
    padding=0
    split=400000
    for i in tqdm(range(max(padding+split*cluster_n,ii_idx),min(len(output_json),padding+split*(cluster_n+1)),1)):
        json_response = solve(output_json[i]['instruction'],output_json[i]['input'])
        cnt = 0
        while cnt < 10 and json_response is None:
            print('trying')
            json_response = solve(output_json[i]['instruction'],output_json[i]['input'])
            cnt += 1
        if cnt < 10:
            json_response['qqid'] = i
            responses_final.append(json_response)
        if (i != 0 and ((i + 1) % 2000 == 0) ) or (i == min(len(output_json),padding+split*(cluster_n+1)) - 1):
            output_json_2 = json.dumps(responses_final,ensure_ascii=False,indent=1)
            with open('../DG_src/dataset/item_prompt_AO/GM_exat_{}_{}.json'.format(i, cluster_n), 'w') as opf:
                opf.write(output_json_2)
            responses_final = []

#print(completion.choices[0].message["content"])

parser = argparse.ArgumentParser(description='Demo of argparse')
 
parser.add_argument('--clustern', type=int, default=0)
parser.add_argument('--ii_idx', type=int, default=0)
args = parser.parse_args()

infer(args.clustern,args.ii_idx)
    
