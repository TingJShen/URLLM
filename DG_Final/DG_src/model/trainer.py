import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import torch_utils
from model.C2DSR import C2DSR
import pdb
import numpy as np

class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")
import torch
import torch.nn as nn

class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class CDSRTrainer(Trainer):
    def __init__(self, opt, adj = None, adj_single = None,adj_att=None,adj_att_source=None,adj_att_target=None):
        self.opt = opt
        if opt["model"] == "C2DSR":
            self.model = C2DSR(opt, adj, adj_single,adj_att,adj_att_source,adj_att_target)
        else:
            print("please select a valid model")
            exit(0)

        self.mi_loss = 0
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        self.MSE_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.CS_criterion = nn.CrossEntropyLoss(reduction='none')
        self.MMD_criterion = MMDLoss(kernel_mul=2.0, kernel_num=5)
        if opt['cuda']:
            self.model.cuda()
            self.BCE_criterion.cuda()
            self.CS_criterion.cuda()
            self.MSE_criterion.cuda()
            self.MMD_criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])

    def get_dot_score(self, A_embedding, B_embedding):
        output = (A_embedding * B_embedding).sum(dim=-1)
        return output

    def unpack_batch_predict(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            X_last = inputs[6]
            Y_last = inputs[7]
            XorY = inputs[8]
            ground_truth = inputs[9]
            neg_list = inputs[10]
            user_idcies = inputs[11]
        else:
            inputs = [Variable(b) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            X_last = inputs[6]
            Y_last = inputs[7]
            XorY = inputs[8]
            ground_truth = inputs[9]
            neg_list = inputs[10]
            user_idcies = inputs[11]
        return seq, x_seq, y_seq, position, x_position, y_position, X_last, Y_last, XorY, ground_truth, neg_list, user_idcies

    def unpack_batch(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            ground = inputs[6]
            share_x_ground = inputs[7]
            share_y_ground = inputs[8]
            x_ground = inputs[9]
            y_ground = inputs[10]
            ground_mask = inputs[11]
            share_x_ground_mask = inputs[12]
            share_y_ground_mask = inputs[13]
            x_ground_mask = inputs[14]
            y_ground_mask = inputs[15]
            corru_x = inputs[16]
            corru_y = inputs[17]
            user_idcies = inputs[18]
        else:
            inputs = [Variable(b) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            ground = inputs[6]
            share_x_ground = inputs[7]
            share_y_ground = inputs[8]
            x_ground = inputs[9]
            y_ground = inputs[10]
            ground_mask = inputs[11]
            share_x_ground_mask = inputs[12]
            share_y_ground_mask = inputs[13]
            x_ground_mask = inputs[14]
            y_ground_mask = inputs[15]
            corru_x = inputs[16]
            corru_y = inputs[17]
            user_idcies = inputs[18]
        return seq, x_seq, y_seq, position, x_position, y_position, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y, user_idcies

    def HingeLoss(self, pos, neg):
        gamma = torch.tensor(self.opt["margin"])
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()


    def train_batch(self, batch,opt):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.graph_convolution()
        self.model.graph_attr_convolution()
        #print('len(batch)',len(batch)) # 19
        seq, x_seq, y_seq, position, x_position, y_position, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y, user_idcies = self.unpack_batch(batch)
        seqs_fea, x_seqs_fea, y_seqs_fea,item_graph_embedding,attr_align,item_graph_embedding_X,attr_align_X,item_graph_embedding_Y,attr_align_Y = self.model(seq, x_seq, y_seq, position, x_position, y_position,user_idcies,opt)

        corru_x_fea = self.model.false_forward(corru_x, position)
        corru_y_fea = self.model.false_forward(corru_y, position)

        x_mask = x_ground_mask.float().sum(-1).unsqueeze(-1).repeat(1,x_ground_mask.size(-1))
        x_mask = 1 / x_mask
        x_mask = x_ground_mask * x_mask # for mean
        x_mask = x_mask.unsqueeze(-1).repeat(1,1,seqs_fea.size(-1))
        r_x_fea = (x_seqs_fea * x_mask).sum(1)

        y_mask = y_ground_mask.float().sum(-1).unsqueeze(-1).repeat(1, y_ground_mask.size(-1))
        y_mask = 1 / y_mask
        y_mask = y_ground_mask * y_mask # for mean
        y_mask = y_mask.unsqueeze(-1).repeat(1,1,seqs_fea.size(-1))
        r_y_fea = (y_seqs_fea * y_mask).sum(1)


        real_x_fea = (seqs_fea * x_mask).sum(1)
        real_y_fea = (seqs_fea * y_mask).sum(1)
        x_false_fea = (corru_x_fea * x_mask).sum(1)
        y_false_fea = (corru_y_fea * y_mask).sum(1)


        real_x_score = self.model.D_X(r_x_fea, real_y_fea) # the cross-domain infomax
        false_x_score = self.model.D_X(r_x_fea, y_false_fea)

        real_y_score = self.model.D_Y(r_y_fea, real_x_fea)
        false_y_score = self.model.D_Y(r_y_fea, x_false_fea)

        pos_label = torch.ones_like(real_x_score).cuda()
        neg_label = torch.zeros_like(false_x_score).cuda()
        x_mi_real = self.BCE_criterion(real_x_score, pos_label)
        x_mi_false = self.BCE_criterion(false_x_score, neg_label)
        x_mi_loss = x_mi_real + x_mi_false

        y_mi_real = self.BCE_criterion(real_y_score, pos_label)
        y_mi_false = self.BCE_criterion(false_y_score, neg_label)
        y_mi_loss = y_mi_real + y_mi_false

        used = opt['used']
        ground = ground[:,-used:]
        ground_mask = ground_mask[:, -used:]
        share_x_ground = share_x_ground[:, -used:]
        share_x_ground_mask = share_x_ground_mask[:, -used:]
        share_y_ground = share_y_ground[:, -used:]
        share_y_ground_mask = share_y_ground_mask[:, -used:]
        x_ground = x_ground[:, -used:]
        x_ground_mask = x_ground_mask[:, -used:]
        y_ground = y_ground[:, -used:]
        y_ground_mask = y_ground_mask[:, -used:]


        share_x_result =  self.model.lin_X(seqs_fea[:,-used:]) # b * seq * X_num
        share_y_result = self.model.lin_Y(seqs_fea[:, -used:])  # b * seq * Y_num
        share_pad_result = self.model.lin_PAD(seqs_fea[:, -used:])  # b * seq * 1
        share_trans_x_result = torch.cat((share_x_result, share_pad_result), dim=-1)
        share_trans_y_result = torch.cat((share_y_result, share_pad_result), dim=-1)


        specific_x_result = self.model.lin_X(seqs_fea[:,-used:] + x_seqs_fea[:, -used:])  # b * seq * X_num
        specific_x_pad_result = self.model.lin_PAD(x_seqs_fea[:, -used:])  # b * seq * 1
        specific_x_result = torch.cat((specific_x_result, specific_x_pad_result), dim=-1)

        specific_y_result = self.model.lin_Y(seqs_fea[:,-used:] + y_seqs_fea[:, -used:])  # b * seq * Y_num
        specific_y_pad_result = self.model.lin_PAD(y_seqs_fea[:, -used:])  # b * seq * 1
        specific_y_result = torch.cat((specific_y_result, specific_y_pad_result), dim=-1)

        x_share_loss = self.CS_criterion(
            share_trans_x_result.reshape(-1, self.opt["source_item_num"] + 1),
            share_x_ground.reshape(-1))  # b * seq
        y_share_loss = self.CS_criterion(
            share_trans_y_result.reshape(-1, self.opt["target_item_num"] + 1),
            share_y_ground.reshape(-1))  # b * seq
        x_loss = self.CS_criterion(
            specific_x_result.reshape(-1, self.opt["source_item_num"] + 1),
            x_ground.reshape(-1))  # b * seq
        y_loss = self.CS_criterion(
            specific_y_result.reshape(-1, self.opt["target_item_num"] + 1),
            y_ground.reshape(-1))  # b * seq

        #align_mse_loss_o = self.MSE_criterion(item_graph_embedding, attr_align)
        if opt['loss_align']=='MSE':
            align_mse_loss_x = self.MSE_criterion(item_graph_embedding_X, attr_align_X)
            align_mse_loss_y = self.MSE_criterion(item_graph_embedding_Y, attr_align_Y)
        elif opt['loss_align']=='MMD':
            a1, b1, c1 = item_graph_embedding_X.size()
            #print(item_graph_embedding_X.size(),item_graph_embedding_X.view(-1,c1).size())
            align_mse_loss_x = self.MMD_criterion(item_graph_embedding_X.view(-1,c1), attr_align_X.view(-1,c1))
            align_mse_loss_y = self.MMD_criterion(item_graph_embedding_Y.view(-1,c1), attr_align_Y.view(-1,c1))
        x_share_loss = (x_share_loss * (share_x_ground_mask.reshape(-1))).mean()
        y_share_loss = (y_share_loss * (share_y_ground_mask.reshape(-1))).mean()
        x_loss = (x_loss * (x_ground_mask.reshape(-1))).mean()
        y_loss = (y_loss * (y_ground_mask.reshape(-1))).mean()

        loss = self.opt["gamma"]*(self.opt["lambda"]*(x_share_loss + y_share_loss + x_loss + y_loss) + (1 - self.opt["lambda"]) * (x_mi_loss + y_mi_loss))+(1-self.opt["gamma"])*(align_mse_loss_x+align_mse_loss_y)

        self.mi_loss += (1 - self.opt["lambda"]) * (x_mi_loss.item() + y_mi_loss.item())
        self.al_loss += (1 - self.opt["gamma"]) * (align_mse_loss_x.item() + align_mse_loss_y.item())
        loss.backward()
        self.optimizer.step()

        return loss.item()
        
    
    def gain_feat(self,batch,opt):
        #print("###gain_feat",batch)
        seq, x_seq, y_seq, position, x_position, y_position, X_last, Y_last, XorY, ground_truth, neg_list, user_idcies = self.unpack_batch_predict(batch)
        #print("###gain_feat_user_idcies",user_idcies[0:10])
        seqs_fea, x_seqs_fea, y_seqs_fea,item_graph_embedding,attr_align,item_graph_embedding_X,attr_align_X,item_graph_embedding_Y,attr_align_Y = self.model(seq, x_seq, y_seq, position, x_position, y_position,user_idcies,opt)
        #cnt=0
        for i in range(len(XorY)):
            if i==0:
                if XorY[i]==0:
                    ret_seqs_fea=seqs_fea[i,-1].data.cpu().numpy()
                    ret_x_seqs_fea=x_seqs_fea[i,X_last[i]].data.cpu().numpy()
                    ret_y_seqs_fea=y_seqs_fea[i,-1].data.cpu().numpy()
                else:
                    ret_seqs_fea=seqs_fea[i,-1].data.cpu().numpy()
                    ret_x_seqs_fea=x_seqs_fea[i,-1].data.cpu().numpy()
                    ret_y_seqs_fea=y_seqs_fea[i,Y_last[i]].data.cpu().numpy()
            else:
                if XorY[i]==0:
                    ret_seqs_fea=np.vstack((ret_seqs_fea,seqs_fea[i,-1].data.cpu().numpy()))
                    ret_x_seqs_fea=np.vstack((ret_x_seqs_fea,x_seqs_fea[i,X_last[i]].data.cpu().numpy()))
                    ret_y_seqs_fea=np.vstack((ret_y_seqs_fea,y_seqs_fea[i,-1].data.cpu().numpy()))
                else:
                    ret_seqs_fea=np.vstack((ret_seqs_fea,seqs_fea[i,-1].data.cpu().numpy()))
                    ret_x_seqs_fea=np.vstack((ret_x_seqs_fea,x_seqs_fea[i,-1].data.cpu().numpy()))
                    ret_y_seqs_fea=np.vstack((ret_y_seqs_fea,y_seqs_fea[i,Y_last[i]].data.cpu().numpy()))

        return ret_seqs_fea,ret_x_seqs_fea,ret_y_seqs_fea

    def test_batch(self, batch,opt):
        seq, x_seq, y_seq, position, x_position, y_position, X_last, Y_last, XorY, ground_truth, neg_list, user_idcies = self.unpack_batch_predict(batch)
        seqs_fea, x_seqs_fea, y_seqs_fea,item_graph_embedding,attr_align,item_graph_embedding_X,attr_align_X,item_graph_embedding_Y,attr_align_Y =self.model(seq, x_seq, y_seq, position, x_position, y_position,user_idcies,opt)
        #print(seq)
        #print(seqs_fea.shape)
        #print(x_seqs_fea.shape)
        #print(y_seqs_fea.shape)
        # 均为batch(4096)*maxlen(30)*256,顺序不变
        #input()
        #print(ground_truth)
        #print(XorY)
        X_pred = []
        candidate_dict={}
        candidate=[] # is a [][]
        Y_pred = []
        range_HR=10000
        for id, fea in enumerate(seqs_fea): # b * s * f
            #print(id)
            #print(fea.shape)
            if XorY[id] == 0:
                share_fea = seqs_fea[id, -1]
                #print("###",id,X_last,X_last[id])
                specific_fea = x_seqs_fea[id, X_last[id]]
                X_score = self.model.lin_X(share_fea + specific_fea)
                #该线性层映射至目标得分，维度为X维度

                #print(X_score.shape)
                X_score=X_score.squeeze(0)
                #print(X_score.shape)
                #input()
                X_score_candidates=np.argsort((-X_score.data.cpu().numpy()))[0:range_HR]
                candidate_dict[id]=X_score_candidates
                #for idx in range(range_HR):

                cur = X_score[ground_truth[id]]
                ##### if use neg_list, it is not full-search.
                score_larger = (X_score > (cur + 0.00001)).data.cpu().numpy()
                #score_larger = (X_score[neg_list[id]] > (cur + 0.00001)).data.cpu().numpy()
                true_item_rank = np.sum(score_larger) + 1
                X_pred.append(true_item_rank)

            else :
                share_fea = seqs_fea[id, -1]
                specific_fea = y_seqs_fea[id, Y_last[id]]
                Y_score = self.model.lin_Y(share_fea + specific_fea).squeeze(0)
                #print(Y_score.shape)
                
                Y_score_candidates=np.argsort((-Y_score.data.cpu().numpy()))[0:range_HR]
                candidate_dict[id]=Y_score_candidates
                cur = Y_score[ground_truth[id]]
                score_larger = (Y_score > (cur + 0.00001)).data.cpu().numpy()
                #score_larger = (Y_score[neg_list[id]] > (cur + 0.00001)).data.cpu().numpy()
                true_item_rank = np.sum(score_larger) + 1
                Y_pred.append(true_item_rank)
        for id in range(len(ground_truth)):
            candidate.append(candidate_dict[id])
        print(candidate[0:5])
        return X_pred, Y_pred, XorY.data.cpu().numpy().tolist(),candidate