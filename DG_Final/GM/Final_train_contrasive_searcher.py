import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
import torch.nn.functional as F
from utils import *
import argparse
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizex,hidden_sizey, output_size,pert_ep):
        super(MLP, self).__init__()
        self.pert_ep=pert_ep
        self.fc1x = nn.Linear(input_size, hidden_sizex)
        self.fc1y = nn.Linear(input_size, hidden_sizey)
        self.fc2x = nn.Linear(hidden_sizex, output_size)
        self.fc2y = nn.Linear(hidden_sizey, output_size)
    def forward(self, x,y):
        x1 = torch.relu(self.fc1x(x))
        y1 = torch.relu(self.fc1y(y))
        
        params_fc1x = self.fc1x.weight
        params_fc1y = self.fc1y.weight
        #print(x1.shape,y1.shape,params_fc1x.shape,params_fc1y.shape)
        tx= torch.matmul(y1,params_fc1y)
        ty= torch.matmul(x1,params_fc1x)
        
        #print(tx.shape,ty.shape)
        outpx = self.fc2x(x1)
        outpy = self.fc2y(y1)
        return tx,ty,x,y,outpx,outpy
def eval(trained_model,eval_length):
    similar_seq=[] # is a[][]
    with open('./train_F.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            record=line.split('\t')
            tmp_list=[]
            for item_idx in range(2,len(record)-1,1):
                item=int(record[item_idx].split('|')[0].replace('\n',''))
                tmp_list.append(item)
            similar_seq.append(tmp_list)

    with open('./test_F.txt','r') as f:
        lines = f.readlines()
    user_representations_testX= np.load('../GM/DG/DGGM_final_test_x_fea.npy')
    user_representations_testY= np.load('../GM/DG/DGGM_final_test_y_fea.npy')
    XORY_test= np.load('../GM/XORY_test.npy')
    with torch.no_grad():
        user_representations_tensorX = torch.tensor(user_representations_testX, dtype=torch.float32).to(device)
        user_representations_tensorY = torch.tensor(user_representations_testY, dtype=torch.float32).to(device)
        XORY_test_tensor=torch.tensor(XORY_test, dtype=torch.float32).to(device)
        tx,ty,x,y,outputsx,outputsy = trained_model(user_representations_tensorX,user_representations_tensorY)
        #print(XORY_test.shape)
        expanded_batch_XORY = torch.tensor(XORY_test_tensor.unsqueeze(1).repeat(1, outputsx.size(1))).to(device)
        probabilities=(1-expanded_batch_XORY)*outputsx+expanded_batch_XORY*outputsy
        matmul_trte = probabilities.cpu().numpy()
    #for line_idx in range(2,3,1):
    HR_list=[]
    CHR_list=[]
    len_list=[]
    ground_truths=[]

    cnt=0
    for line_idx in tqdm(range(len(lines))):
        line=lines[line_idx]
        tmp_line=line.split('\t')
        tmp_seq_list=[]
        for case in range(2,len(tmp_line)-1,1):
            case_id=int(tmp_line[case].split('|')[0].replace('\n',''))
            if((tmp_line[case].split('|')[1].replace('\n',''))!=(tmp_line[1])):
                tmp_seq_list.append(case_id)
            else:
                ground_truth=int(tmp_line[case].split('|')[0].replace('\n',''))
                ground_truths.append(ground_truth)
        argsort_trte=np.argsort(-matmul_trte[line_idx])
        if matmul_trte[line_idx][argsort_trte[0]]>0.5:
            cnt+=1
        full_seq=[]
        for ii in range(0,eval_length,1):
            full_seq+=similar_seq[argsort_trte[ii]]
        len_list.append(len(full_seq))
        if ground_truth in full_seq:
            HR_list.append(1)
            if matmul_trte[line_idx][argsort_trte[0]]>0.5:
                CHR_list.append(1)
        else:
            HR_list.append(0)
            if matmul_trte[line_idx][argsort_trte[0]]>0.5:
                CHR_list.append(0)
    print(cnt,'CHR:',np.mean(CHR_list))
    ground_truths=np.array(ground_truths)

    HR=np.mean(HR_list)
    mean_len=np.mean(len_list)
    return matmul_trte,cnt, HR, mean_len

def neg_loss(ta,a,neg_size,loss_func,eps=0.5):
    loss=0
    for each_neg in range(neg_size):
        permuted_indices = torch.randperm(a.size(0))
        shuffle_a = a[permuted_indices]
        loss=loss+torch.nn.functional.relu(eps -loss_func(ta,shuffle_a))
    loss=loss/neg_size
    return loss

def train(user_representationsX,user_representationsY, ground_truths,XORY,eval_length,alpha=1,hidden_dimx=256,neg_sample_size=32,hidden_dimy=256, batch_size=128, num_epochs=500, learning_rate=0.001):
    assert neg_sample_size<=batch_size
    input_size = len(user_representationsX[0])
    output_size = len(ground_truths[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_representationsX_tensor = torch.tensor(user_representationsX, dtype=torch.float32).to(device)
    user_representationsY_tensor = torch.tensor(user_representationsY, dtype=torch.float32).to(device)
    ground_truths_tensor = torch.tensor(ground_truths, dtype=torch.float32).to(device)
    XORY_tensor = torch.tensor(XORY, dtype=torch.float32).to(device)
    model = MLP(input_size, hidden_dimx,hidden_dimy, output_size,0.05).to(device)
    BCE_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    MSE_crit = torch.nn.MSELoss(reduction="mean")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    min_loss=9999
    for epoch in range(num_epochs):
        for i in range(0, len(user_representationsX_tensor), batch_size):
            batch_user_representationsX = user_representationsX_tensor[i:i+batch_size]
            batch_user_representationsY = user_representationsY_tensor[i:i+batch_size]

            batch_ground_truths = ground_truths_tensor[i:i+batch_size]
            batch_XORY=XORY_tensor[i:i+batch_size]
            tx,ty,x,y,outputsx,outputsy = model(batch_user_representationsX,batch_user_representationsY)
            loss_orth = MSE_crit(tx,x)+ MSE_crit(ty,y)+neg_loss(tx,x,neg_sample_size,MSE_crit)+neg_loss(ty,y,neg_sample_size,MSE_crit)
            expanded_batch_XORY = torch.tensor(batch_XORY.unsqueeze(1).repeat(1, outputsx.size(1))).to(device)
            outputs=(1-expanded_batch_XORY)*outputsx+expanded_batch_XORY*outputsy
            loss_crit = criterion(outputs, batch_ground_truths)
            loss= (1-alpha)*loss_orth+alpha*loss_crit
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}, Loss_crit: {loss_crit.item():.8f}, Loss_orth: {loss_orth.item():.8f}')
        if (epoch+1) % 10 == 0:
            #eval
            if loss.item()<min_loss:
                best_hr=HR
                best_l=mean_len
                best_trte=matmul_trte
                best_hrl=HR/mean_len
            print(f'Epoch [{epoch+1}/{num_epochs}], cnt: {cnt}, HR: {HR}, mean_len: {mean_len}')
        #if loss < 1e-8:
        #    break

    return best_hr,best_hrl,best_l,best_trte
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A program that takes three arguments.')
    parser.add_argument('--arg1', type=int, help='First argument')
    parser.add_argument('--arg2', type=int, help='First argument')
    parser.add_argument('--arg3', type=int, help='First argument')
    args = parser.parse_args()
    super_parameter=[args.arg1,args.arg2,args.arg3]
    print(super_parameter)

    final_minlen_user=initialize_with_embedding_tr_390(super_parameter[0],super_parameter[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp_X=np.load('../GM/DG/DGGM_final_train_x_fea.npy')
    user_representationsX = np.load('../GM/DG/DGGM_final_train_x_fea.npy')
    user_representationsY = np.load('../GM/DG/DGGM_final_train_y_fea.npy')
    ground_truths = np.zeros((len(user_representationsX),len(tmp_X)),dtype=int)
    for idx in range(len(final_minlen_user)):
        ground_truths[final_minlen_user[idx][0]][final_minlen_user[idx][1]]=1
    XORY=np.load('../GM/XORY_train.npy')
    print(user_representationsX.shape)
    print(ground_truths.shape)

    hidden_units_set=328
    neg_sample_size=16
    GNND=328
    result_HR,result_HRL,result_L,best_trte=train(user_representationsX,user_representationsY,\
                        ground_truths,XORY,alpha=0.9,hidden_dimx=hidden_units_set,neg_sample_size=16,hidden_dimy=GNND,eval_length=neg_sample_size)
    np.save('./saver/best_trte_XORY_DG_.npy',max_best_trte)




    

