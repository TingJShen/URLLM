import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import torch_utils, helper
from utils.GraphMaker import GraphMaker
from utils.GraphMaker_attribute import GraphMaker_attribute
from model.trainer import CDSRTrainer
from utils.loader import *
import json
import codecs
from tqdm import tqdm
import pandas as pd
import pdb


            

def train(hidden_units,hidden_units_attr,GNND,user_len,GAC,loss_align):
    parser = argparse.ArgumentParser()
    # dataset part
    parser.add_argument('--data_dir', type=str, default='Food-Kitchen', help='Movie-Book, Entertainment-Education')

    # model part
    parser.add_argument('--model', type=str, default="DG", help='model name')
    parser.add_argument('--hidden_units', type=int, default=hidden_units, help='lantent dim.')
    parser.add_argument('--hidden_units_attr', type=int, default=hidden_units_attr, help='lantent_attr dim.')
    parser.add_argument('--num_blocks', type=int, default=2, help='lantent dim.')
    parser.add_argument('--num_heads', type=int, default=1, help='lantent dim.')
    parser.add_argument('--GNN', type=int, default=GNND, help='GNN depth.')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate.')
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                        help='Optimizer: sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
    parser.add_argument('--lr_decay', type=float, default=1, help='Learning rate decay rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--leakey', type=float, default=0.1)
    parser.add_argument('--maxlen', type=int, default=15)
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--lambda', type=float, default=0.7)
    parser.add_argument('--gamma', type=float, default=0.7)

    # train part
    parser.add_argument('--num_epoch', type=int, default=15, help='Number of total training epochs.')
    parser.add_argument('--batch_size', type=int, default=384, help='Training batch size.')
    parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
    parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
    parser.add_argument('--save_epoch', type=int, default=1, help='Save model checkpoints every k epochs.')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
    parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
    parser.add_argument('--seed', type=int, default=2040)
    parser.add_argument('--load', dest='load', action='store_true', default=False,  help='Load pretrained model.')
    parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
    parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
    parser.add_argument('--undebug', action='store_false', default=True)
    parser.add_argument('--augmenting', type=str, default='', help='Optional info for the experiment.')

    def seed_everything(seed=1111):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args = parser.parse_args()
    print(args)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)
    init_time = time.time()
    # make opt
    opt = vars(args)

    seed_everything(opt["seed"])

    model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
    model_save_dir = opt['save_dir'] + '/' + model_id
    opt['model_save_dir'] = model_save_dir
    opt['used']=user_len
    opt['gamma']=0.5
    opt['GAC']=GAC

    opt['loss_align']=loss_align
    opt['pertubed']=1
    opt['eps_pertubed']=0.05


    helper.ensure_dir(model_save_dir, verbose=True)
    # save config
    helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
    file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'],
                                    header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

    # print model info
    helper.print_config(opt)

    if opt["undebug"]:
        pass
        # opt["cuda"] = False
        # opt["cpu"] = True
    filename = opt["data_dir"]
    train_data = "./dataset/" + filename + "/traindata_new.txt"

    line_count=0
    train_user_num=0
    test_user_num=0
    valid_user_num=0
    with open("./dataset/" + filename + "/traindata_new.txt", 'r') as file:
        for line in file:
            train_user_num += 1
            line_count += 1
    #print('##line_count',line_count)
    with open("./dataset/" + filename + "/testdata_new.txt", 'r') as file:
        for line in file:
            test_user_num += 1
            line_count += 1
    #print('##line_count',line_count)
    with open("./dataset/" + filename + "/validdata_new.txt", 'r') as file:
        for line in file:
            valid_user_num += 1
            line_count += 1
    ##print('##line_count',line_count)

    opt["train_user_num"] = train_user_num
    opt["test_user_num"] = test_user_num
    opt["valid_user_num"] = valid_user_num
    opt["usernum"] = line_count

    print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
    train_batch = DataLoader(opt['data_dir'], opt['batch_size'], opt, evaluation = -1)
    print("##train_batch",len(train_batch))
    train_batch_not_shuffled = DataLoader(opt['data_dir'], opt['batch_size'], opt, evaluation = 3)
    valid_batch = DataLoader(opt['data_dir'], opt["batch_size"], opt, evaluation = 2)
    print(len(valid_batch))
    test_batch = DataLoader(opt['data_dir'], opt["batch_size"], opt, evaluation = 1)
    print(len(test_batch))
    print("Data loading done!")

    opt["itemnum"] = opt["source_item_num"] + opt["target_item_num"] + 1
    print("opt[itemnum]",opt["itemnum"])



    G = GraphMaker(opt, train_data)
    if filename == 'Entertainment-Education_AO':
        if args.augmenting=='N':
            G_2 = GraphMaker_attribute(opt,'./dataset/user_prompt_AO','AO','')
    if filename == 'Entertainment-Education_Amazon':
        G_2 = GraphMaker_attribute(opt,'./user_prompt_GM','GM','./item_prompt_GM')
    #print('#@@2',G)
    adj, adj_single = G.adj, G.adj_single
    adj_att, adj_att_source,adj_att_target = G_2.adj_att, G_2.adj_att_source, G_2.adj_att_target
    opt["attributenum"]=G_2.attribute_len
    #print('#@@1',adj, adj_single,adj_att, adj_att_source,adj_att_target)
    print("graph loaded!")
    #input()


    if opt["cuda"]:
        adj = adj.cuda()
        adj_single = adj_single.cuda()
        adj_att = adj_att.cuda()
        adj_att_source = adj_att_source.cuda()
        adj_att_target = adj_att_target.cuda()



    # model
    if not opt['load']:
        trainer = CDSRTrainer(opt, adj, adj_single,adj_att,adj_att_source,adj_att_target)
    else:
        exit(0)

    global_step = 0
    current_lr = opt["lr"]
    format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/epoch), lr: {:.6f}'
    num_batch = len(train_batch)
    max_steps = opt['num_epoch'] * num_batch

    print("Start training:")

    begin_time = time.time()
    X_dev_score_history=[0]
    Y_dev_score_history=[0]
    def read_hr(dataset,matmul_trte):
        if args.data_dir == 'Entertainment-Education_Amazon':  
            path_GM='../GM'
            similar_seq=[] # is a[][]
            case=open(path_GM+'/train_F.txt','r')
            lines = case.readlines()
            for line in lines:
                record=line.split('\t')
                tmp_list=[]
                for item_idx in range(2,len(record)-1,1):
                    item=int(record[item_idx].split('|')[0].replace('\n',''))
                    tmp_list.append(item)
                similar_seq.append(tmp_list) 
            path_GM='../GM'
            case=open(path_GM+'/test_F.txt','r')
            lines = case.readlines()
            case.close()
        if args.data_dir == 'Entertainment-Education_AO':
            path_AO='../AO/'
            similar_seq=[] # is a[][]
            case=open(path_AO+'/train_F2.txt','r')
            lines = case.readlines()
            case.close()
            for line in lines:
                record=line.split('\t')
                tmp_list=[]
                for item_idx in range(2,len(record)-1,1):
                    item=int(record[item_idx].split('|')[0].replace('\n',''))
                    tmp_list.append(item)
                similar_seq.append(tmp_list)
            path_AO='../AO/'
            case=open(path_AO+'/test_F2.txt','r')
            lines = case.readlines()
            case.close()
        #for line_idx in range(2,3,1):
        HR_list=[]
        len_list=[]
        #print('####',lines[0],len(lines))
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
            argsort_trte=np.argsort(matmul_trte[line_idx])
            full_seq=[]
            for ii in range(0,1,1):
                full_seq+=similar_seq[argsort_trte[ii]]

            len_list.append(len(full_seq))
            if ground_truth in full_seq:
                HR_list.append(1)
            else:
                HR_list.append(0)

        HR=np.mean(HR_list)
        mean_len=np.mean(len_list)
        print("#readrc+mean",mean_len)
        print("#readrc+HR",HR)        
        return HR




    if args.data_dir == 'Entertainment-Education_Amazon':
        data_rec = pd.read_csv('../GM/item_listM_F.csv',sep=',',header=0).values[0::,0::]
    if args.data_dir == 'Entertainment-Education_AO':
        data_rec = pd.read_csv('../AO/item_listA_F.csv',sep=',',header=0).values[0::,0::]
    post_items=[]
    for item in data_rec:
        post_items.append(item[0])
    M_length=len(post_items)
    def read_candidate(dataset,test_candidate,test_XorY):
        ground_truth=[]

        HR=[1,5,10,20]
        if args.data_dir == 'Entertainment-Education_Amazon':
            f=open('../GM/test_F.txt','r')
        if args.data_dir == 'Entertainment-Education_AO':
            f=open('../AO/test_F2.txt','r')
        lines=f.readlines()
        f.close()
        for line in lines:
            item_list=line.split('\t')
            ans_time=item_list[1]
            #print(ans_time)
            for i in range(2,len(item_list),1):
                rec_list=item_list[i].split('|')
                if len(rec_list)>1:
                    if rec_list[1]==ans_time:
                        ground_truth.append(int(rec_list[0]))
        #print('gt',ground_truth[0:50])
        for i in range(test_candidate.shape[0]):
            if test_XorY[i] != 0:
                for j in range(test_candidate.shape[1]):
                    test_candidate[i][j]+=M_length
        ans=np.zeros((3,len(HR)),dtype='float')  
        for num in range(2):
            for each_HR_idx in range(len(HR)):
                each_HR=HR[each_HR_idx]
                acc=0
                all=0
                for i in range(len(ground_truth)):
                    if test_XorY[i] == num:
                        if ground_truth[i] in test_candidate[i][0:each_HR]:
                            acc+=1
                        all+=1
                #print(each_HR,acc,all,acc/all)
                ans[num,each_HR_idx]=acc/all
        for each_HR_idx in range(len(HR)):
            each_HR=HR[each_HR_idx]
            acc=0
            all=0
            for i in range(len(ground_truth)):
                if ground_truth[i] in test_candidate[i][0:each_HR]:
                    acc+=1
                all+=1
            #print(each_HR,acc,all,acc/all)
            ans[2,each_HR_idx]=acc/all
        #print(ans)

        def mean_reciprocal_rank(recommendations, ground_truth):
            """
            计算Mean Reciprocal Rank (MRR)

            Parameters:
            - recommendations: 一个包含推荐结果的列表，每个元素是一个排名列表
            - ground_truth: 一个包含每个查询的正确答案的列表，每个元素是一个包含正确答案的排名列表

            Returns:
            - mrr: MRR 分数
            """
            reciprocal_ranks = []
            for recs, truth in zip(recommendations, ground_truth):
                try:
                    # 查找正确答案的排名
                    rank = np.where(recs == truth)[0][0] + 1  # 假设1表示正确答案，索引加1为排名
                    reciprocal_ranks.append(1 / rank)
                except IndexError:
                    reciprocal_ranks.append(0.0)  # 如果正确答案不在排名中，则添加0

            # 计算MRR
            if not reciprocal_ranks:
                return 0.0
            mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

            return mrr

        # 例子
        '''
        recommendations = [
            [3, 1, 2, 4],  # 推荐结果的排名
            [1, 2, 3, 4],
            [4, 2, 3, 1],
        ]

        ground_truth = [2, 1, 3]  # 每个查询的正确答案
        '''
        # 计算MRR
        mrr_score = mean_reciprocal_rank(np.array(test_candidate), ground_truth)
        #print(f"MRR: {mrr_score}")
        return ans,mrr_score




    max_mrr_score=0
    max_HR2=0
    for epoch in range(1, opt['num_epoch'] + 1):
        train_loss = 0
        epoch_start_time = time.time()
        trainer.mi_loss = 0
        trainer.al_loss = 0
        for batch in train_batch:
            #print(len(batch))
            global_step += 1
            loss = trainer.train_batch(batch,opt)
            train_loss += loss

        duration = time.time() - epoch_start_time
        print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                                        opt['num_epoch'], train_loss/num_batch, duration, current_lr))
        print("mi:", trainer.mi_loss/num_batch)
        print("al:", trainer.al_loss,trainer.al_loss/num_batch)

        #if epoch % 5:
        #    continue

        # eval model
        print("Evaluating on dev set...")

        trainer.model.eval()
        trainer.model.graph_convolution()


        def cal_test_score(predictions):
            MRR=0.0
            HR_1 = 0.0
            HR_5 = 0.0
            HR_10 = 0.0
            NDCG_5 = 0.0
            NDCG_10 = 0.0
            valid_entity = 0.0
            # pdb.set_trace()
            #print(len(predictions))
            for pred in predictions:
                valid_entity += 1
                MRR += 1 / pred
                if pred <= 1:
                    HR_1 += 1
                if pred <= 5:
                    NDCG_5 += 1 / np.log2(pred + 1)
                    HR_5 += 1
                if pred <= 10:
                    NDCG_10 += 1 / np.log2(pred + 1)
                    HR_10 += 1
                #if valid_entity % 100 == 0:
                    #print('.', end='')
            if valid_entity==0:
                print('valid:0!')
                return 0,0,0,0,0,0
            return MRR/valid_entity, NDCG_5 / valid_entity, NDCG_10 / valid_entity, HR_1 / valid_entity, HR_5 / valid_entity, HR_10 / valid_entity
        '''
        def get_full_feat(evaluation_batch):
            for i, batch in enumerate(evaluation_batch):
                seqs_fea,x_seqs_fea, y_seqs_fea=trainer.gain_feat(batch,opt)
                if i==0:
                    all_seqs_fea=seqs_fea.data.cpu().numpy()
                    all_x_seqs_fea=x_seqs_fea.data.cpu().numpy()
                    all_y_seqs_fea=y_seqs_fea.data.cpu().numpy()
                else:
                    all_seqs_fea=np.vstack((all_seqs_fea,seqs_fea.data.cpu().numpy()))
                    all_x_seqs_fea=np.vstack((all_x_seqs_fea,x_seqs_fea.data.cpu().numpy()))
                    all_y_seqs_fea=np.vstack((all_y_seqs_fea,y_seqs_fea.data.cpu().numpy()))
            print(all_seqs_fea.shape,all_x_seqs_fea.shape,all_y_seqs_fea.shape)
            return all_seqs_fea,all_x_seqs_fea,all_y_seqs_fea
        '''
        def get_full_feat(evaluation_batch):
            for i, batch in enumerate(evaluation_batch):
                seqs_fea,x_seqs_fea, y_seqs_fea=trainer.gain_feat(batch,opt)
                if i==0:
                    all_seqs_fea=seqs_fea
                    all_x_seqs_fea=x_seqs_fea
                    all_y_seqs_fea=y_seqs_fea
                else:
                    all_seqs_fea=np.vstack((all_seqs_fea,seqs_fea))
                    all_x_seqs_fea=np.vstack((all_x_seqs_fea,x_seqs_fea))
                    all_y_seqs_fea=np.vstack((all_y_seqs_fea,y_seqs_fea))
            print(all_seqs_fea.shape,all_x_seqs_fea.shape,all_y_seqs_fea.shape)
            return all_seqs_fea,all_x_seqs_fea,all_y_seqs_fea
        
        '''
        def get_full_scores(evaluation_batch):
            for i, batch in enumerate(evaluation_batch):
                x_seqs_fea, y_seqs_fea=trainer.gain_feat(batch,,opt)
                if i==0:
                    all_x_seqs_fea=x_seqs_fea.data.cpu().numpy()
                    all_y_seqs_fea=y_seqs_fea.data.cpu().numpy()
                else:
                    all_x_seqs_fea=np.vstack((all_x_seqs_fea,x_seqs_fea.data.cpu().numpy()))
                    all_y_seqs_fea=np.vstack((all_y_seqs_fea,y_seqs_fea.data.cpu().numpy()))
            print(all_x_seqs_fea.shape,all_y_seqs_fea.shape)
            return all_x_seqs_fea,all_y_seqs_fea
        '''
        def get_evaluation_result(evaluation_batch):
            debug=0
            X_pred = []
            Y_pred = []
            candidate=[]
            XorY = []
            for i, batch in enumerate(evaluation_batch):
                X_predictions, Y_predictions,XorY_batch,candidate_batch = trainer.test_batch(batch,opt)
                X_pred = X_pred + X_predictions
                Y_pred = Y_pred + Y_predictions
                XorY = XorY + XorY_batch
                candidate=candidate+candidate_batch
            # 找到最长的子数组长度
            max_length = max(len(subarray) for subarray in candidate)

            # 创建一个形状为 (len(original_array), max_length) 的新数组，用 0 填充
            padded_array = np.zeros((len(candidate), max_length), dtype=int)

            # 将原始数组的值复制到新数组中
            for i, subarray in enumerate(candidate):
                padded_array[i, :len(subarray)] = subarray
            candidate=padded_array
            if debug==1:
                tmp_list=[]
                for item in candidate:
                    print(len(item))
                #print(candidate)
            candidate=np.array(candidate)
            print('@@#@',len(XorY),candidate.shape)
            return X_pred, Y_pred, XorY, candidate

        def get_similarity(featx,featy,similarity_type):
            if similarity_type=='cos':
                featx=featx/np.linalg.norm(featx,axis=1,keepdims=True)
                featy=featy/np.linalg.norm(featy,axis=1,keepdims=True)
                similarity=-np.matmul(featx,featy.T)
            if similarity_type=='dot':
                similarity=-np.matmul(featx,featy.T)
            if similarity_type=='l2':
                similarity = np.linalg.norm(featx[:, np.newaxis, :] - featy, axis=2)
            if similarity_type=='l2_norm_1':
                featx=featx/np.linalg.norm(featx,axis=1,keepdims=True)
                featy=featy/np.linalg.norm(featy,axis=1,keepdims=True)
                similarity = np.linalg.norm(featx[:, np.newaxis, :] - featy, axis=2)
            if similarity_type=='l2_norm_0':
                featx=featx/np.linalg.norm(featx,axis=0,keepdims=True)
                featy=featy/np.linalg.norm(featy,axis=0,keepdims=True)
                similarity = np.linalg.norm(featx[:, np.newaxis, :] - featy, axis=2)
            
            return similarity


        val_X_pred, val_Y_pred , val_XorY, val_candidate= get_evaluation_result(valid_batch)
        val_X_MRR, val_X_NDCG_5, val_X_NDCG_10, val_X_HR_1, val_X_HR_5, val_X_HR_10 = cal_test_score(val_X_pred)
        val_Y_MRR, val_Y_NDCG_5, val_Y_NDCG_10, val_Y_HR_1, val_Y_HR_5, val_Y_HR_10 = cal_test_score(val_Y_pred)

        print("")
        print('val epoch:%d, time: %f(s), X (MRR: %.4f, NDCG@10: %.4f, HR@10: %.4f), Y (MRR: %.4f, NDCG@10: %.4f, HR@10: %.4f)'
            % (epoch, time.time() - begin_time, val_X_MRR, val_X_NDCG_10, val_X_HR_10, val_Y_MRR, val_Y_NDCG_10, val_Y_HR_10))


        if (val_X_MRR > max(X_dev_score_history) or val_Y_MRR > max(Y_dev_score_history)):
            test_X_pred, test_Y_pred , test_XorY, test_candidate= get_evaluation_result(test_batch)
            
            test_seqs_feas,test_x_seqs_feas, test_y_seqs_feas=get_full_feat(test_batch)
            train_seqs_feas,train_x_seqs_feas, train_y_seqs_feas=get_full_feat(train_batch_not_shuffled)
            test_x_fea=test_seqs_feas+test_x_seqs_feas
            test_y_fea=test_seqs_feas+test_y_seqs_feas
            train_x_fea=train_seqs_feas+train_x_seqs_feas
            train_y_fea=train_seqs_feas+train_y_seqs_feas

            print('test_x_fea',test_x_fea.shape)
            print('test_y_fea',test_y_fea.shape)
            matmul_trte_x=get_similarity(test_x_fea,train_x_fea,'dot')
            matmul_trte_y=get_similarity(test_y_fea,train_y_fea,'dot')
            matmul_trte=np.zeros(matmul_trte_x.shape)
            for i in tqdm(range(matmul_trte_x.shape[0])):
                if test_XorY[i]==0:
                    matmul_trte[i,:]=matmul_trte_x[i,:]
                else:
                    matmul_trte[i,:]=matmul_trte_y[i,:]
            
            final_test_candidate=test_candidate
            final_matmul_trte=matmul_trte
            print('#####',len(test_X_pred))
            print("#####",len(test_Y_pred))
            test_X_MRR, test_X_NDCG_5, test_X_NDCG_10, test_X_HR_1, test_X_HR_5, test_X_HR_10 = cal_test_score(test_X_pred)
            test_Y_MRR, test_Y_NDCG_5, test_Y_NDCG_10, test_Y_HR_1, test_Y_HR_5, test_Y_HR_10 = cal_test_score(test_Y_pred)
            if val_Y_MRR > max(Y_dev_score_history):
                print("X best!")
                print([test_X_MRR, test_X_NDCG_5, test_X_NDCG_10, test_X_HR_1, test_X_HR_5, test_X_HR_10])

            if val_Y_MRR > max(Y_dev_score_history):
                print("Y best!")
                print([test_Y_MRR, test_Y_NDCG_5, test_Y_NDCG_10, test_Y_HR_1, test_Y_HR_5, test_Y_HR_10])

            X_dev_score_history.append(val_X_MRR)
            Y_dev_score_history.append(val_Y_MRR)
    return final_matmul_trte,final_test_candidate


hidden_units_set=[328,320]
hidden_units_attr_set=[27,28,29]
GNND_set=[1]
gammas=[0.5]
eps=[0.05]
user_len=[9,10,11]
result_HR=np.zeros((len(hidden_units_set),len(hidden_units_attr_set),len(user_len)),dtype='float')
result_MRR=np.zeros((len(hidden_units_set),len(hidden_units_attr_set),len(user_len)),dtype='float')
max_HR=0
max_MRR=0
for hidden_units in range(len(hidden_units_set)):
    for hidden_units_attr in range(len(hidden_units_attr_set)):
        for gamma1 in range(len(user_len)):
            print('##',hidden_units,hidden_units_attr,gamma1)
            result_HR[hidden_units][hidden_units_attr][gamma1],result_MRR[hidden_units][hidden_units_attr][gamma1],tmp_matmul_trte,tmp_test_candidate=train(hidden_units_set[hidden_units],hidden_units_attr_set[hidden_units_attr],1,user_len[gamma1],1,'MSE')
            if result_HR[hidden_units][hidden_units_attr][gamma1]>max_HR:
                max_HR=result_HR[hidden_units][hidden_units_attr][gamma1]
                result_matmul_trte=tmp_matmul_trte
            if result_MRR[hidden_units][hidden_units_attr][gamma1]>max_MRR:
                max_MRR=result_MRR[hidden_units][hidden_units_attr][gamma1]
                result_test_candidate=tmp_test_candidate
            np.save('t4_G2'+'_final_'+'DG'+'result_matmul_trte.npy',result_matmul_trte)
            np.save('t4_G2'+'_final_'+'DG'+'result_test_candidate.npy',result_test_candidate)
            print("###result!!!HR",result_HR)
            print("###result!!!MRR",result_MRR)
