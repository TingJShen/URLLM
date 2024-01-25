import numpy as np
import torch
import torch.nn as nn
from model.GNN import GCNLayer


class Discriminator(torch.nn.Module):
    def __init__(self, n_in,n_out):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_in, n_out, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, S, node, s_bias=None):
        score = self.f_k(node, S)
        if s_bias is not None:
            score += s_bias
        return score

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class ATTENTION(torch.nn.Module):
    def __init__(self, opt):
        super(ATTENTION, self).__init__()
        self.opt = opt
        self.emb_dropout = torch.nn.Dropout(p=self.opt["dropout"])
        self.pos_emb = torch.nn.Embedding(self.opt["maxlen"], self.opt["hidden_units"]+self.opt["hidden_units"], padding_idx=0)  # TO IMPROVE
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.opt["hidden_units"]+self.opt["hidden_units"], eps=1e-8)

        for _ in range(self.opt["num_blocks"]):
            new_attn_layernorm = torch.nn.LayerNorm(self.opt["hidden_units"]+self.opt["hidden_units"], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(self.opt["hidden_units"]+self.opt["hidden_units"],
                                                            self.opt["num_heads"],
                                                            self.opt["dropout"])
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.opt["hidden_units"]+self.opt["hidden_units"], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.opt["hidden_units"]+self.opt["hidden_units"], self.opt["dropout"])
            self.forward_layers.append(new_fwd_layer)

    def forward(self, seqs_data, seqs, position):
        # positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(position)
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(seqs_data.cpu() == self.opt["itemnum"] - 1)
        if self.opt["cuda"]:
            timeline_mask = timeline_mask.cuda()
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool))
        if self.opt["cuda"]:
            attention_mask = attention_mask.cuda()

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

class DG(torch.nn.Module):
    def __init__(self, opt, adj, adj_single,adj_att,adj_att_source,adj_att_target):
        super(DG, self).__init__()
        self.opt = opt
        #print(opt)
        '''bin/{'data_dir': 'Entertainment-Education_AO', 'model': 'DG', 'hidden_units': 256, 'num_blocks': 2, 'num_heads': 1, 
        'GNN': 1, 'dropout': 0.2, 'optim': 'adam', 'lr': 0.001, 'lr_decay': 1, 'weight_decay': 0.0005, 'decay_epoch': 5, 'max_grad_norm': 5.0, 
        'leakey': 0.1, 'maxlen': 30, 'cpu': False, 'cuda': True, 'lambda': 0.7, 'num_epoch': 4, 'batch_size': 128, 'log_step': 200, 'log': 'logs.txt', 
        'save_epoch': 1, 'save_dir': './saved_models', 'id': '00', 'seed': 2040, 'load': False, 'model_file': None, 'info': '', 'undebug': True, 
        'model_save_dir': './saved_models/00', 'source_item_num': 18639, 'target_item_num': 19757, 'itemnum': 38397}'''
        #input()
        self.item_emb_X = torch.nn.Embedding(self.opt["itemnum"], self.opt["hidden_units"],
                                           padding_idx=self.opt["itemnum"] - 1)
        self.item_emb_Y = torch.nn.Embedding(self.opt["itemnum"], self.opt["hidden_units"],
                                           padding_idx=self.opt["itemnum"] - 1)
        self.item_emb = torch.nn.Embedding(self.opt["itemnum"], self.opt["hidden_units"],
                                           padding_idx=self.opt["itemnum"] - 1)

        #self.item_user_attr_emb_att_X = torch.nn.Embedding(self.opt["itemnum"]+self.opt["usernum"]+self.opt["attributenum"], self.opt["hidden_units_attr"],
        #                                   padding_idx=self.opt["itemnum"] - 1)
        #self.item_user_attr_emb_att_Y = torch.nn.Embedding(self.opt["itemnum"]+self.opt["usernum"]+self.opt["attributenum"], self.opt["hidden_units_attr"],
        #                                   padding_idx=self.opt["itemnum"] - 1)
        self.item_user_attr_emb_att = torch.nn.Embedding(self.opt["itemnum"]+self.opt["usernum"]+self.opt["attributenum"], self.opt["hidden_units_attr"],
                                           padding_idx=self.opt["itemnum"] - 1)

        self.GNN_encoder_X = GCNLayer(opt,0)
        self.GNN_encoder_Y = GCNLayer(opt,0)
        self.GNN_encoder = GCNLayer(opt,0)

        #self.GNN_encoder_X_attr = GCNLayer(opt,1)
        #self.GNN_encoder_Y_attr = GCNLayer(opt,1)
        self.GNN_encoder_attr = GCNLayer(opt,1)
        self.adj = adj
        self.adj_single = adj_single
        self.adj_att=adj_att
        self.adj_att_source=adj_att_source
        self.adj_att_target=adj_att_target

        self.D_X = Discriminator(self.opt["hidden_units"]+self.opt["hidden_units"], self.opt["hidden_units"]+self.opt["hidden_units"])
        self.D_Y = Discriminator(self.opt["hidden_units"]+self.opt["hidden_units"], self.opt["hidden_units"]+self.opt["hidden_units"])

        self.lin_X = nn.Linear(self.opt["hidden_units"]+self.opt["hidden_units"], self.opt["source_item_num"])
        self.lin_Y = nn.Linear(self.opt["hidden_units"]+self.opt["hidden_units"], self.opt["target_item_num"])
        self.lin_PAD = nn.Linear(self.opt["hidden_units"]+self.opt["hidden_units"], 1)
        self.lin_ALIGN = nn.Linear(self.opt["hidden_units_attr"],self.opt["hidden_units"])
        self.lin_ALIGN_X = nn.Linear(self.opt["hidden_units_attr"],self.opt["hidden_units"])
        self.lin_ALIGN_Y = nn.Linear(self.opt["hidden_units_attr"],self.opt["hidden_units"])
        self.encoder = ATTENTION(opt)
        self.encoder_X = ATTENTION(opt)
        self.encoder_Y = ATTENTION(opt)

        self.source_item_index = torch.arange(0, self.opt["source_item_num"], 1)
        self.target_item_index = torch.arange(self.opt["source_item_num"], self.opt["source_item_num"]+self.opt["target_item_num"], 1)
        self.item_index = torch.arange(0, self.opt["itemnum"], 1)
        self.item_user_att_index = torch.arange(0, self.opt["itemnum"]+self.opt["usernum"]+self.opt["attributenum"], 1)
        if self.opt["cuda"]:
            self.source_item_index = self.source_item_index.cuda()
            self.target_item_index = self.target_item_index.cuda()
            self.item_index = self.item_index.cuda()
            self.item_user_att_index = self.item_user_att_index.cuda()

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def graph_attr_convolution(self):
        item_emb_att_fea = self.my_index_select_embedding(self.item_user_attr_emb_att, self.item_user_att_index)
        #item_emb_att_fea_X = self.my_index_select_embedding(self.item_user_attr_emb_att_X, self.item_user_att_index)
        #item_emb_att_fea_Y = self.my_index_select_embedding(self.item_user_attr_emb_att_Y, self.item_user_att_index)
        self.cross_emb_att = self.GNN_encoder_attr(item_emb_att_fea, self.adj_att)
        #self.single_emb_att_X = self.GNN_encoder_X_attr(item_emb_att_fea_X, self.adj_att_source)
        #self.single_emb_att_Y = self.GNN_encoder_Y_attr(item_emb_att_fea_Y, self.adj_att_target)

    def graph_convolution(self):
        fea = self.my_index_select_embedding(self.item_emb, self.item_index)
        #print("self.item_emb",self.item_emb.weight.shape,"self.item_index",self.item_index,"fea",fea,fea.data.shape)
        #input()
        fea_X = self.my_index_select_embedding(self.item_emb_X, self.item_index)
        #print("self.item_emb_X",self.item_emb_X,"self.item_index",self.item_index,"fea_X",fea_X,fea_X.shape)
        fea_Y = self.my_index_select_embedding(self.item_emb_Y, self.item_index)

        self.cross_emb = self.GNN_encoder(fea, self.adj)
        self.single_emb_X = self.GNN_encoder_X(fea_X, self.adj_single)
        self.single_emb_Y = self.GNN_encoder_Y(fea_Y, self.adj_single)

    def forward(self, o_seqs, x_seqs, y_seqs, position, x_position, y_position,user_indices,opt):
        #TODO: seq_item?
        #print("####1",self.cross_emb, o_seqs)
        #print("####2",self.my_index_select(self.cross_emb, o_seqs).shape,self.item_emb(o_seqs).shape)
        #print("####2",self.my_index_select(self.cross_emb, o_seqs).shape,self.item_emb(o_seqs).shape) #torch.Size([128, 30, 256]) torch.Size([128, 30, 256])
        item_graph_embedding=self.my_index_select(self.cross_emb, o_seqs) + self.item_emb(o_seqs)
        attr_align=self.lin_ALIGN(self.my_index_select(self.cross_emb_att, o_seqs))+self.lin_ALIGN(self.item_user_attr_emb_att(o_seqs))
        seqs = torch.cat((item_graph_embedding,attr_align),dim=-1)
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs_fea = self.encoder(o_seqs, seqs, position)
        #print("user_indices",user_indices)
        #print("user_indices+itemnum",(user_indices+opt['itemnum']))
        '''
        users_emb =self.my_index_select(self.cross_emb_att, (user_indices+opt['itemnum']))
        users_emb *= self.item_emb.embedding_dim ** 0.5
        users_X_emb = self.my_index_select(self.single_emb_att_X, user_indices+opt['itemnum'])
        users_X_emb *= self.item_emb.embedding_dim ** 0.5
        users_Y_emb = self.my_index_select(self.single_emb_att_Y, user_indices+opt['itemnum'])
        users_Y_emb *= self.item_emb.embedding_dim ** 0.5
        '''
        #print(x_seqs,y_seqs) # no stack!
        #seqs = self.my_index_select(self.single_emb_X, x_seqs) + self.item_emb_X(x_seqs)
        item_graph_embedding_X=self.my_index_select(self.single_emb_X, x_seqs) + self.item_emb_X(x_seqs)
        attr_align_X=self.lin_ALIGN_X(self.my_index_select(self.cross_emb_att, x_seqs))+self.lin_ALIGN(self.item_user_attr_emb_att(x_seqs))
        seqs = torch.cat((item_graph_embedding_X,attr_align_X),dim=-1)
        seqs *= self.item_emb.embedding_dim ** 0.5
        x_seqs_fea = self.encoder_X(x_seqs, seqs, x_position)

        #seqs = self.my_index_select(self.single_emb_Y, y_seqs) + self.item_emb_Y(y_seqs)
        item_graph_embedding_Y=self.my_index_select(self.single_emb_Y, y_seqs) + self.item_emb_Y(y_seqs)
        attr_align_Y=self.lin_ALIGN_Y(self.my_index_select(self.cross_emb_att, y_seqs))+self.lin_ALIGN(self.item_user_attr_emb_att(y_seqs))
        seqs = torch.cat((item_graph_embedding_Y,attr_align_Y),dim=-1)
        seqs *= self.item_emb.embedding_dim ** 0.5
        y_seqs_fea = self.encoder_Y(y_seqs, seqs, y_position)

        #print('seqs_fea.shape',seqs_fea.shape,'x_seqs_fea.shape',x_seqs_fea.shape,'y_seqs_fea.shape',y_seqs_fea.shape,'users_emb.shape',users_emb.shape,'users_X_emb.shape',users_X_emb.shape,'users_Y_emb.shape',users_Y_emb.shape)
        #input()
        return seqs_fea, x_seqs_fea, y_seqs_fea,item_graph_embedding,attr_align,item_graph_embedding_X,attr_align_X,item_graph_embedding_Y,attr_align_Y

    def false_forward(self, false_seqs, position):
        self.item_graph_embedding=self.my_index_select(self.cross_emb, false_seqs) + self.item_emb(false_seqs)
        self.attr_align=self.lin_ALIGN(self.my_index_select(self.cross_emb_att, false_seqs))+self.lin_ALIGN(self.item_user_attr_emb_att(false_seqs))
        seqs = torch.cat((self.item_graph_embedding,self.attr_align),dim=-1)
        if opt['pertubed'] == 1:
            seqs += torch.sign(seqs)*torch.randn(seqs.shape).cuda() * opt['eps_pertubed']
        #seqs = torch.cat((self.my_index_select(self.cross_emb, false_seqs) + self.item_emb(false_seqs),self.my_index_select(self.cross_emb_att, false_seqs) + self.item_user_attr_emb_att(false_seqs)),dim=-1)
        seqs *= self.item_emb.embedding_dim ** 0.5
        false_seqs_fea = self.encoder(false_seqs, seqs, position)
        return false_seqs_fea