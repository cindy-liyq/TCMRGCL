import torch
import torch.nn as nn
from utils import sparse_dropout,normal_adj_matrix
import numpy as np
import torch.nn.functional as F
import time
import psutil
import random
import scipy.sparse as sp
from scipy.sparse import csr_matrix,coo_matrix
from random import shuffle,randint,choice
from tqdm import tqdm
import pandas as pd
seed = 2023
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
mem=psutil.virtual_memory()

class SHGCN(nn.Module):

    def __init__(self, dim):
        super(SHGCN, self).__init__()
        self.lin = torch.nn.Linear(dim, dim)
        self.tanh = torch.nn.Tanh()

    def forward(self, input, adj):
        x = self.lin(torch.spmm(adj.to(torch.float),input))
        return self.tanh(x)

class GCL_Encoder(nn.Module):
    def __init__(self, adj,dim,drop_rate, temp,device):
        super(GCL_Encoder, self).__init__()
        self.drop_rate = drop_rate  # 0.1
        self.dim = dim  # 64
        self.n_layers = 2  # 2
        self.temp = temp  # 0.2
        self.svd_q=5
        self.adj=adj
        self.num_sym = self.adj.shape[0]
        self.num_herb = self.adj.shape[1]
        self.device=device
        self.embedding_dict = self._init_model()  # 初始化user_emb和item_emb


    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'sym_emb': nn.Parameter(initializer(torch.empty(self.num_sym, self.dim))),
            'herb_emb': nn.Parameter(initializer(torch.empty(self.num_herb, self.dim))),
        })
        return embedding_dict

    def SVD_graph_augment(self):
        num_s,num_h = self.num_sym , self.num_herb
        n_nodes = num_s+num_h
        svd_u, s, svd_v = torch.svd_lowrank(self.adj, q=self.svd_q)
        u_mul_s = svd_u @ (torch.diag(s))
        v_mul_s = svd_v @ (torch.diag(s))
        svd_s = u_mul_s @ svd_v.T  # svd_s为(360,753)
        svd_h = v_mul_s @ svd_u.T  # svd_h为(753,360)

        adj_mat = np.zeros((n_nodes,n_nodes),dtype=np.float32)
        adj_mat[:num_s,num_s:]=svd_s.cpu().numpy()
        adj_mat[num_s:,:num_s]=svd_h.cpu().numpy()
        adj_mat = csr_matrix(adj_mat)
        adj_mat.data[np.isinf(adj_mat.data)] = 0.
        return self.convert_sparse_mat_to_tensor(adj_mat).cuda()

    def graph_reconstruction(self,aug_type):
        if aug_type==0:
            dropped_adj=self.SVD_graph_augment()
        elif aug_type==1:#aug_type=1
            dropped_adj = self.random_graph_augment(aug_type)#随机构建增强图
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def edge_dropout(self,adj, drop_rate):
        """Input: a sparse user-item adjacency matrix and a dropout rate."""
        adj_shape = adj.get_shape()  # (360,7533)
        edge_count = adj.count_nonzero()  # 35000
        row_idx, col_idx = adj.nonzero()  # col_idx:35000个数的数组---列，row_idx:35000个数的数组---行
        keep_idx = random.sample(range(edge_count), int(edge_count * (1 - drop_rate)))  # 去除部分边后得到保留的边
        sym_np = np.array(row_idx)[keep_idx]
        herb_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(sym_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (sym_np, herb_np)), shape=adj_shape)#(360,753)
        return dropped_adj

    '''
    def node_dropout(self,adj, drop_rate):
        """Input: a sparse adjacency matrix and a dropout rate."""
        adj_shape = adj.get_shape()
        row_idx, col_idx = adj.nonzero()
        drop_user_idx = random.sample(range(adj_shape[0]), int(adj_shape[0] * drop_rate))
        drop_item_idx = random.sample(range(adj_shape[1]), int(adj_shape[1] * drop_rate))
        indicator_user = np.ones(adj_shape[0], dtype=np.float32)
        indicator_item = np.ones(adj_shape[1], dtype=np.float32)
        indicator_user[drop_user_idx] = 0.
        indicator_item[drop_item_idx] = 0.
        diag_indicator_user = sp.diags(indicator_user)
        diag_indicator_item = sp.diags(indicator_item)
        mat = sp.csr_matrix(
            (np.ones_like(row_idx, dtype=np.float32), (row_idx, col_idx)),
            shape=(adj_shape[0], adj_shape[1]))
        mat_prime = diag_indicator_user.dot(mat).dot(diag_indicator_item)
        return mat_prime
    '''

    def convert_sparse_mat_to_tensor(self,X):  # 转化为coo格式的稀疏矩阵
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)


    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()#(360,753)
        n_nodes = adj_shape[0]+adj_shape[1]#1113
        (sym_np_keep, herb_np_keep) = adj_mat.nonzero()#dropout之后剩下的边只剩644251条
        ratings_keep = np.ones_like(adj_mat.data)
        tmp_adj = sp.csr_matrix((ratings_keep, (sym_np_keep, herb_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        tmp_adj = coo_matrix(tmp_adj)
        # return self.normalize_graph_mat(tmp_adj)
        return normal_adj_matrix(tmp_adj)

    def random_graph_augment(self,aug_type):
        dropped_mat = None
        interaction_mat = csr_matrix(self.adj.cpu().numpy())
        if aug_type == 0:
            dropped_mat = self.node_dropout(interaction_mat, self.drop_rate)#通过随机丢弃节点进行重构图
        elif aug_type == 1 or self.aug_type == 2:
            dropped_mat = self.edge_dropout(interaction_mat, self.drop_rate)#通过随机丢弃边进行重构图,丢弃了10%的边
        dropped_mat = self.convert_to_laplacian_mat(dropped_mat)#把新的邻接矩阵重新归一化
        return self.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def InfoNCE(self,view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        s_idx = torch.unique(torch.Tensor(idx[0]))
        h_idx = torch.unique(torch.Tensor(idx[1]))
        sym_view_1, herb_view_1 = self.forward(perturbed_mat1)
        sym_view_2, herb_view_2 = self.forward(perturbed_mat2)
        view1 = torch.cat((sym_view_1[s_idx],herb_view_1[h_idx]),0)
        view2 = torch.cat((sym_view_2[s_idx],herb_view_2[h_idx]),0)
        # user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        # item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        #return user_cl_loss + item_cl_loss
        return self.InfoNCE(view1,view2,self.temp)


    def forward(self,perturbed_adj=None):
        ego_embeddings = torch.cat([self.embedding_dict['sym_emb'], self.embedding_dict['herb_emb']],
                                   0)  # (1113,256)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj, list):  # 这段代码的含义是，如果 perturbed_adj 是一个列表，那么就执行
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)  # 矩阵相乘
            else:
                ego_embeddings = torch.sparse.mm(self.convert_sparse_mat_to_tensor(self.convert_to_laplacian_mat(csr_matrix(self.adj.cpu().numpy()))).cuda(), ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings,
                                     dim=1)  # 把三层的嵌入堆叠在一起(39642,3,64)********************tip2------换聚合方式
        all_embeddings = torch.mean(all_embeddings,
                                    dim=1)  # 取平均作为最终的嵌入(39642,64)*************************tip2------换聚合方式
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.num_sym, self.num_herb])
        return user_all_embeddings, item_all_embeddings


class Prompts_Generator(nn.Module):
    def __init__(self, emb_size, prompt_size):
        super(Prompts_Generator, self).__init__()

        self.layers = nn.ModuleList(
            # [nn.Linear(emb_size, prompt_size), nn.Linear(prompt_size, prompt_size)])#3333333333333old  # emb_size=64,prompt_size=256
            [nn.Linear(emb_size, prompt_size)])#3333333333333new  # emb_size=64,prompt_size=256
        self.activation = nn.Tanh()
        # self.activation = nn.Sigmoid()

    def forward(self, inputs):
        prompts = inputs
        for i in range(len(self.layers)):
            prompts = self.layers[i](prompts)
            prompts = self.activation(prompts)

        return prompts


class Fusion_MLP(nn.Module):
    def __init__(self, emb_size, prompt_size):
        super(Fusion_MLP, self).__init__()

        self.layers = nn.ModuleList(
            # [nn.Linear(emb_size+emb_size + prompt_size, emb_size), nn.Linear(emb_size, emb_size)])#4444444444444444444old  # 嵌入特征从320(64+256)变化为64
            # [nn.Linear(emb_size+emb_size + prompt_size, emb_size), nn.Linear(emb_size, emb_size)])#4444444444444444444new old # 嵌入特征从320(64+256)变化为64
            [nn.Linear(emb_size + prompt_size, emb_size)])#4444444444444444444new new # 嵌入特征从320(64+256)变化为64
        self.activation = nn.Tanh()

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activation(x)

        return x

class FRGCL(nn.Module):  # input_dim=64,hidden_dim=256,output_dim=256
    def __init__(self, num_s, num_h, adj_sh, adj_ss, adj_hh, dim, layer, temp, lambda_1,
                 lambda_2, dropout, batch, device,grand_herb, grand_sym,
                 grand1_layer, grand2_layer, grand3_layer,epoch,origin_data):
        super(FRGCL, self).__init__()
        # 1.定义函数对节点进行随机初始化表示
        self.e_s0 = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_s, dim))).to(device)  # sh图中的s节点初始化
        self.e_h0 = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_h, dim))).to(device) # sh图中的h节点初始化
        self.e_ss0 = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_s, dim))).to(device)  # ss图中的s节点初始化
        self.e_hh0 = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_h, dim))).to(device)  # hh图中的h节点初始化

        self.E_s_list = [[] for i in range(layer + 1)]
        self.E_h_list = [[] for i in range(layer + 1)]
        self.E_s_list[0] = self.e_s0
        self.E_h_list[0] = self.e_h0

        self.Z_s_list = [[] for i in range(layer + 1)]
        self.Z_h_list = [[] for i in range(layer + 1)]
        self.Z_s_list[0] = self.e_s0
        self.Z_h_list[0] = self.e_h0

        self.E_ss_list = [[] for i in range(layer + 1)]
        self.E_hh_list = [[] for i in range(layer + 1)]
        self.E_ss_list[0] = self.e_ss0
        self.E_hh_list[0] = self.e_hh0

        self.grand_layer = [grand1_layer,grand2_layer,grand3_layer]

        # 2.构建针对SH和SSHH图分别构建GCN模块，用于邻居聚合和消息更新
        self.sh_gcn = SHGCN(dim)
        self.ss_gcn = SHGCN(dim)
        self.hh_gcn = SHGCN(dim)

        self.prompt_size = 256

        #预训练模块
        self.pre_train_model = GCL_Encoder(adj_sh,dim,dropout, temp,device)#包含了症状的初始embed和中药得初始embed
        self.prompts_generator = Prompts_Generator(dim, self.prompt_size).to(device)#两个线性层，两个激活函数
        self.fusion_mlp=Fusion_MLP(dim, self.prompt_size).to(device)#两个线性层。两个激活函数


        self.grand_herb = grand_herb
        self.grand_sym = grand_sym

        self.adj_sh = adj_sh
        self.adj_ss = adj_ss
        self.adj_hh = adj_hh

        self.num_s = num_s
        self.num_h = num_h
        self.dim = dim
        self.layer = layer
        self.n_layers = 2
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.batch = batch
        self.epoch=epoch

        self.SI_bn = torch.nn.BatchNorm1d(dim)
        self.relu = torch.nn.ReLU(inplace=False)

        self.device = device

        self.bestPerformance = []
        self.bestPerformance_data = []


        self.topK = [5, 10, 15, 20]

        self.origin_data=origin_data
        self.reg = 0.0001


    # 对每个节点进行邻居聚合
    def neighbor_aggregation(self, x_list,grand_x):
        aggregated_embeddings = []
        for i, ids in enumerate(grand_x):
            # 获取对应频度的节点应聚合的邻居层数
            layers_to_aggregate = self.grand_layer[i]
            # 为当前频度的节点聚合embedding
            # 初始聚合为第一个元素（初始embedding）
            agg_embedding = x_list[0][ids]
            # 根据需要聚合的邻居层数，累加更多层的embedding
            for layer in range(1, layers_to_aggregate + 1):
                agg_embedding += x_list[layer][ids]/(torch.norm(x_list[layer][ids], p=2, dim=1).view(-1, 1))
            # 将聚合后的embedding添加到结果列表
            aggregated_embeddings.append(agg_embedding/(layers_to_aggregate + 1))
        return aggregated_embeddings

    # 整合三个梯度矩阵
    def conform_grade_matrix(self, grand_feature, num_x,grand_x):
        # 整合症状矩阵
        combined_features = torch.zeros((num_x, self.dim))
        for i, x_list in enumerate(grand_x):
            for j, xid in enumerate(x_list):
                combined_features[int(xid)] = grand_feature[i][j]
        return combined_features

    # 计算相似度矩阵
    def cosine_similarity(self, x, y):
        """
            get the cosine similarity between to matrix
            consin(x, y) = xy / (sqrt(x^2) * sqrt(y^2))
        """
        x = x - torch.mean(x)
        y = y - torch.mean(y)
        xy = torch.matmul(x, y.transpose(0, 1))
        x_norm = torch.sqrt(torch.mul(x, x).sum(1))
        y_norm = torch.sqrt(torch.mul(y, y).sum(1))
        x_norm = 1.0 / (x_norm.unsqueeze(1) + 1e-8)
        y_norm = 1.0 / (y_norm.unsqueeze(0) + 1e-8)
        # xy = torch.mul(torch.mul(xy, x_norm), y_norm)
        l = 5
        num_b = x.shape[0] // l
        if num_b * l < x.shape[0]:
            l = l + 1
        result = xy.clone()  # 创建一个新的张量来存储中间结果
        for i in range(l):
            begin = i * num_b
            end = (i + 1) * num_b
            end = xy.shape[0] if end > xy.shape[0] else end
            result[begin:end] = torch.mul(torch.mul(xy[begin:end], x_norm[begin:end]), y_norm)
        return result

    def next_batch_pairwise(self,data, batch_size):
        training_data = data.training_data
        shuffle(training_data)  # 先打乱，然后按顺序取batch
        batch_id = 0
        data_size = len(training_data)
        while batch_id < data_size:
            if batch_id + batch_size <= data_size:
                users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
                items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
                batch_id += batch_size
            else:
                users = [training_data[idx][0] for idx in range(batch_id, data_size)]
                items = [training_data[idx][1] for idx in range(batch_id, data_size)]
                batch_id = data_size
            u_idx, i_idx, j_idx = [], [], []
            item_list = list(data.item.keys())
            for i, user in enumerate(users):
                i_idx.append(data.item[items[i]])
                u_idx.append(data.user[user])
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
            yield u_idx, i_idx, j_idx




    def _pre_train(self):
        """
        一个通过SVD重构的矩阵，一个经过dropout随机丢弃后的矩阵，做对比学习，得出user和item的embedding,预训练不涉及推荐损失，只是为了得到更好的user_embed和item_embed
        :return:
        """
        maxPreEpoch=10
        pre_trained_model = self.pre_train_model.to(self.device)  # SGL_Encoder
        optimizer = torch.optim.Adam(pre_trained_model.parameters(), lr=0.01)
        print('############## Pre-Training Phase ##############')
        for epoch in range(maxPreEpoch):
            dropped_adj1 = pre_trained_model.graph_reconstruction(aug_type=0)#(1113,1113)#svd矩阵分解得到的重构矩阵
            dropped_adj2 = pre_trained_model.graph_reconstruction(aug_type=1)

            self.origin_data['train_loader'].dataset.neg_sampling()
            # batch_num = len(self.origin_data['train_loader'])
            # batch_no = int(np.ceil(self.origin_data['num_Ptrain'] / batch_num))
            for n, batch in enumerate(self.origin_data['train_loader']):
                sids, pos, neg = batch
                sids = sids.long().to(self.device)
                pos = pos.long().to(self.device)
                neg = neg.long().to(self.device)
                cl_loss = pre_trained_model.cal_cl_loss([sids, pos], dropped_adj1, dropped_adj2)
                batch_loss=cl_loss
                optimizer.zero_grad()

                if epoch == self.epoch-1:
                    batch_loss.backward(retain_graph=True)
                else:
                    batch_loss.backward()
                optimizer.step()
                if n % 100==0:
                    print('pre-training:', epoch + 1, 'batch', n, 'cl_loss', cl_loss.item())




    def fenceng_aggregation(self,):
        # 原视图SH图中

        # self.E_s_list[0]=sym_emb#2222222222222222222old
        self.E_s_list[0]=self.e_s0#2222222222222222222new
        # self.E_h_list[0]=herb_emb#2222222222222222222old
        self.E_h_list[0]=self.e_h0#2222222222222222222new
        for l in range(1, self.layer + 1):
            self.E_s_list[l] = self.sh_gcn(self.E_h_list[l - 1], self.adj_sh)
            self.E_h_list[l] = self.sh_gcn(self.E_s_list[l - 1], self.adj_sh.T)

        E_s_norm = self.neighbor_aggregation(self.E_s_list, self.grand_sym)
        e_s = self.conform_grade_matrix(E_s_norm, self.num_s, self.grand_sym).to(self.device)
        E_h_norm = self.neighbor_aggregation(self.E_h_list, self.grand_herb)
        e_h = self.conform_grade_matrix(E_h_norm, self.num_h, self.grand_herb).to(self.device)


        # 4.从主视图SH、SS和HH症状s和中药h的信息聚合成最终的节点表示
        g_s = (e_s / (torch.norm(e_s, p=2, dim=1).view(-1, 1)))
        g_h = (e_h / (torch.norm(e_h, p=2, dim=1).view(-1, 1)))

        # 5.由症状节点和中药节点和处方训练集得出处方对中药的预测
        return g_s,g_h

    def _high_order_u_relations(self,adj_ss,adj_hh):
        # sym_embeddings=sym_emb#11111111111111111111 old
        sym_embeddings=self.e_ss0#11111111111111111111 new
        # herb_embeddings=herb_emb#11111111111111111111111111 old
        herb_embeddings=self.e_hh0#11111111111111111111 new
        sym_all_embeddings = [sym_embeddings]
        herb_all_embeddings = [herb_embeddings]
        for k in range(1):
            sym_embeddings = torch.mm(adj_ss, sym_embeddings)
            sym_all_embeddings.append(sym_embeddings)

            herb_embeddings = torch.sparse.mm(adj_hh, herb_embeddings)
            herb_all_embeddings.append(herb_embeddings)

        sym_all_embeddings = torch.stack(sym_all_embeddings, dim=1)
        sym_all_embeddings = torch.mean(sym_all_embeddings, dim=1)

        herb_all_embeddings = torch.stack(herb_all_embeddings, dim=1)
        herb_all_embeddings = torch.mean(herb_all_embeddings, dim=1)

        return sym_all_embeddings,herb_all_embeddings

    def _prompts_generation(self,adj_ss,adj_hh):
        sym_embed,herb_embed = self._high_order_u_relations(adj_ss,adj_hh)
        prompts_sym = self.prompts_generator(sym_embed)
        prompts_herb = self.prompts_generator(herb_embed)

        return prompts_sym,prompts_herb

    def _prompts_u_embeddings_fusion(self, prompts_sym,prompts_herb,fenceng_sym_embedding,fenceng_herb_embedding):
        # prompts_sym_emb = torch.cat((prompts_sym, sym_emb,fenceng_sym_embedding), 1)#55555555555555555555old
        prompts_sym_emb = torch.cat((prompts_sym, fenceng_sym_embedding), 1)#55555555555555555555new
        # prompts_herb_emb = torch.cat((prompts_herb, herb_emb,fenceng_herb_embedding), 1)#55555555555555555555old
        prompts_herb_emb = torch.cat((prompts_herb, fenceng_herb_embedding), 1)#55555555555555555555new
        prompted_sym_emb = self.fusion_mlp(prompts_sym_emb)
        prompts_herb_emb = self.fusion_mlp(prompts_herb_emb)
        return prompted_sym_emb,prompts_herb_emb

    def bpr_loss(self,sym_emb, pos_herb_emb, neg_herb_emb):
        pos_score = torch.mul(sym_emb, pos_herb_emb).sum(dim=1)
        neg_score = torch.mul(sym_emb, neg_herb_emb).sum(dim=1)
        loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
        return torch.mean(loss)


    def l2_reg_loss(self,reg,*args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2)
        return emb_loss * reg

    def predict(self,prescription,sym_emb,herb_emb):
        pres_s = torch.mm(prescription, sym_emb)
        sum_pre = prescription.sum(dim=1).view(-1, 1)
        pres_s = pres_s / sum_pre
        pres_s = self.SI_bn(pres_s)
        pres_s = self.relu(pres_s)
        pres_h = torch.mm(pres_s, herb_emb.T)
        return pres_h

    def ranking_evaluation(self,prediction_result,test_measure):
        # 对预测结果得到降序排列后的元素索引

        sorted_prediction = np.argsort(-prediction_result.cpu().detach().numpy(), axis=1)
        for i, (hid, sid) in enumerate(zip(self.ph.cpu(), self.ps.cpu())):
            syms = np.where(sid == 1)[0].tolist()
            self.output_syms.append(syms)

            trueLabel = np.where(hid == 1)[0].tolist()
            self.output_groundtruth.append(trueLabel)
            result = sorted_prediction[i]

            # ************定性结果分析
            predict_herbs = result[:len(trueLabel)].tolist()
            self.output_predictherbs.append(predict_herbs)  # 真实处方取多少个，我的预测处方就取多少个
            matched_herb = set(trueLabel) & set(predict_herbs)
            count = len(matched_herb)
            self.output_precision.append([count / len(trueLabel)])
            # ************定性结果分析

            for i in range(len(self.topK)):
                topk = result[:self.topK[i]]
                matched_herb = set(trueLabel) & set(topk)
                count = len(matched_herb)
                test_measure[0][i]+= count / self.topK[i]
                test_measure[1][i]+= count / len(trueLabel)
        measure = test_measure
        return measure

    def test_rec(self,current_sym_emb,current_herb_emb):
        self.output_syms = []
        self.output_groundtruth = []
        self.output_predictherbs = []
        self.output_precision = []
        def f1_score(measure):
            for i in range(len(self.topK)):
                measure[2][i] = 2 * measure[0][i] * measure[1][i] / (measure[0][i] + measure[1][i])
            return measure
        batch_num = len(self.origin_data['test_loader'])
        batch_no = int(np.ceil(self.origin_data['num_Ptest'] / batch_num))
        test_measure = np.zeros((3, 4), dtype=np.float32)
        for j, test_batch in enumerate(self.origin_data['test_loader']):
            _, _, _ = test_batch
            start = j * batch_no
            end = min((j + 1) * batch_no, self.origin_data['num_Ptest'])
            ps = self.origin_data['ps_test_onehot'][start:end]
            ph = self.origin_data['ph_test_onehot'][start:end]
            self.ps = torch.tensor(ps).to(self.device)
            self.ph = torch.tensor(ph).to(self.device)
            prediction_result = self.predict(self.ps,current_sym_emb,current_herb_emb)
            measure = self.ranking_evaluation(prediction_result,test_measure)
        measure/=self.origin_data['num_Ptest']
        measure = f1_score(measure)
        output_syms = np.array(self.output_syms, dtype=object).reshape(-1, 1)
        output_groundtruth = np.array(self.output_groundtruth, dtype=object).reshape(-1, 1)
        output_predictherbs = np.array(self.output_predictherbs, dtype=object).reshape(-1, 1)
        output_precision = np.array(self.output_precision, dtype=object).reshape(-1, 1)
        output_result = np.concatenate((output_syms, output_groundtruth, output_predictherbs, output_precision), axis=1)
        return measure,output_result

    # def save(self):
    #     best_sym_emb, best_herb_emb = self.pre_train_model.forward()
    #     prompts_sym, prompts_herb = self._prompts_generation(best_herb_emb, best_sym_emb, self.adj_ss, self.adj_hh)
    #     # 分层结构的GCN进行邻居聚合
    #     fenceng_sym_embedding, fenceng_herb_embedding = self.fenceng_aggregation(best_sym_emb,best_herb_emb)
    #     prompted_sym_emb, prompted_herb_emb = self._prompts_u_embeddings_fusion(prompts_sym, best_sym_emb,
    #                                                                              prompts_herb, best_herb_emb,
    #                                                                              fenceng_sym_embedding,
    #                                                                              fenceng_herb_embedding)
    #     # 最终的user和item的embedding
    #     self.best_sym_emb = prompted_sym_emb
    #     self.best_herb_emb = prompted_herb_emb

    def fast_evaluation(self,epoch,bestPerformance,bestPerformance_data,current_sym_emb,current_herb_emb,best_sym_emb,best_herb_emb):
        print('evaluating the model...')
        measure,output_result = self.test_rec(current_sym_emb,current_herb_emb)

        if len(bestPerformance) > 0:
            count = 0
            mean_values = [0, 0]
            performance = {}
            performance['Precision'] = measure[0]
            performance['Recall'] = measure[1]
            performance['F1-score'] = measure[2]
            for (key1, value1), (key2, value2) in zip(bestPerformance[1].items(), performance.items()):
                mean_values[0] = np.mean(value1)
                mean_values[1] = np.mean(value2)
                if mean_values[0] < mean_values[1]:
                    count += 1
                else:
                    count -= 1

            if count > 0:
                bestPerformance[1] = performance
                bestPerformance[0] = epoch+1
                bestPerformance_data[1] = output_result
                bestPerformance_data[0] = epoch + 1
                # self.save()
                best_sym_emb = current_sym_emb
                best_herb_emb = current_herb_emb

        else:
            bestPerformance.append(epoch + 1)
            bestPerformance_data.append(epoch + 1)
            performance = {}
            performance['Precision']=measure[0]
            performance['Recall']=measure[1]
            performance['F1-score']=measure[2]
            bestPerformance.append(performance)
            bestPerformance_data.append(output_result)
            best_sym_emb = current_sym_emb
            best_herb_emb = current_herb_emb
            # self.save()
        print('-' * 120)
        print('Quick Ranking Performance ' + ' Item Recommendation)')
        bp=''
        bp+='Precision' + ':' + str(measure[0]) + ' | '
        bp += 'Recall' + ':' + str(measure[1]) + ' | '
        bp += 'F1-score' + ':' + str(measure[2]) + ' | '
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', bp)


        bp = ''
        bp += 'Precision' + ':' + str(bestPerformance[1]['Precision']) + ' | '
        bp += 'Recall' + ':' + str(bestPerformance[1]['Recall']) + ' | '
        bp += 'F1-score' + ':' + str(bestPerformance[1]['F1-score']) + ' | '
        print('*Best Performance* ')
        print('Epoch:', str(bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return best_sym_emb,best_herb_emb

    # 输入一个稀疏矩阵adj
    def train(self,optimizer):
        bestPerformance=[]
        bestPerformance_data=[]
        best_sym_emb = None
        best_herb_emb = None
        for epoch in range(self.epoch):
            self.origin_data['train_loader'].dataset.neg_sampling()
            batch_num = len(self.origin_data['train_loader'])
            batch_no = int(np.ceil(self.origin_data['num_Ptrain'] / batch_num))
            criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
            for n, batch in enumerate(self.origin_data['train_loader']):

                prompts_sym,prompts_herb = self._prompts_generation(self.adj_ss,self.adj_hh)
                fenceng_sym_embedding,fenceng_herb_embedding=self.fenceng_aggregation()
                # todo:整合embedding
                prompted_sym_emb,prompted_herb_emb = self._prompts_u_embeddings_fusion(prompts_sym,prompts_herb,fenceng_sym_embedding,fenceng_herb_embedding)

                sids, pos, neg = batch
                sids = sids.long().to(self.device)
                pos = pos.long().to(self.device)
                neg = neg.long().to(self.device)
                start = n * batch_no
                end = min((n + 1) * batch_no, self.origin_data['num_Ptrain'])
                ps = self.origin_data['ps_train_onehot'][start:end]
                ph = self.origin_data['ph_train_onehot'][start:end]
                ps = torch.tensor(ps).to(self.device)
                ph = torch.tensor(ph).to(self.device)

                rec_sym_emb, rec_herb_emb = prompted_sym_emb, prompted_herb_emb
                prediction_result=self.predict(ps,rec_sym_emb,rec_herb_emb)
                pre_loss = criterion(prediction_result, ph)
                sym_emb, pos_herb_emb, neg_herb_emb = rec_sym_emb[sids], rec_herb_emb[pos], rec_herb_emb[
                    neg]
                rec_loss = self.bpr_loss(sym_emb, pos_herb_emb, neg_herb_emb)
                batch_loss = rec_loss + self.l2_reg_loss(self.reg, sym_emb, pos_herb_emb)+pre_loss
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print('training:', epoch + 1, 'batch', n+1, 'rec_loss:', rec_loss.item(),'pre_loss:',pre_loss.item())

            # with torch.no_grad():
            #     sym_emb, herb_emb = model()
            #     prompts_sym,prompts_herb = self._prompts_generation(herb_emb, sym_emb,self.adj_ss,self.adj_hh)
            #     # 分层结构的GCN进行邻居聚合
            #     fenceng_sym_embedding, fenceng_herb_embedding = self.fenceng_aggregation(sym_emb,herb_emb)
            #     prompted_sym_emb, prompted_herb_emb = self._prompts_u_embeddings_fusion(prompts_sym, sym_emb,
            #                                                                              prompts_herb, herb_emb,
            #                                                                              fenceng_sym_embedding,
            #                                                                              fenceng_herb_embedding)
                #最终的user和item的embedding
                current_sym_emb = prompted_sym_emb
                current_herb_emb = prompted_herb_emb

            # todo:可视化best_sym_emb和best_herb_emb
            best_sym_emb,best_herb_emb=self.fast_evaluation(epoch,bestPerformance,bestPerformance_data,current_sym_emb,current_herb_emb,best_sym_emb,best_herb_emb)
            #利用pickle将两个embedding打包，我这边可以直接读取embeding
            import pickle
            with open("tsne/tsne/best_noCPreT_sym_emb.pkl", 'wb') as file_1:
                pickle.dump(best_sym_emb,file_1)
            with open("tsne/tsne/best_noCPreT_herb_emb.pkl", 'wb') as file_2:
                pickle.dump(best_herb_emb,file_2)
        # self.sym_emb, self.herb_emb = self.best_sym_emb, self.best_herb_emb
        # self.sym_emb, self.herb_emb = best_sym_emb, best_herb_emb

        print('Testing...')
        bp = ''
        bp += 'Precision' + ':' + str(bestPerformance[1]['Precision']) + ' | '
        bp += 'Recall' + ':' + str(bestPerformance[1]['Recall']) + ' | '
        bp += 'F1-score' + ':' + str(bestPerformance[1]['F1-score']) + ' | '
        print('*Best Performance* ')
        print('Epoch:', str(bestPerformance[0]) + ',', bp)
        data_result = pd.DataFrame(bestPerformance_data[1], columns=['syms', 'groundtruth', 'predictherbs', 'output_precision'])
        data_result.to_excel('output_result.xlsx')












        '''
            # 梯度1邻居信息聚合
            e_s_grand1 = self.gat_layer(self.grand1_layer, self.sh_gcn, self.e_h0, self.e_s0[self.grand_sym[0]],
                                        (self.adj_sh * self.sh_weight)[self.grand_sym[0]])
            e_s_grand1_norm = self.neighbor_aggregation(e_s_grand1)
            e_h_grand1 = self.gat_layer(self.grand1_layer, self.sh_gcn, self.e_s0,
                                        (self.adj_sh.T * self.sh_weight.T)[self.grand_herb[0]])
            e_h_grand1_norm = self.neighbor_aggregation(e_h_grand1)

            # 梯度2邻居信息聚合、
            e_s_grand2 = self.gat_layer(self.grand2_layer, self.sh_gcn, self.e_h0,
                                        (self.adj_sh * self.sh_weight)[self.grand_sym[1]])
            e_s_grand2_norm = self.neighbor_aggregation(e_s_grand2)
            e_h_grand2 = self.gat_layer(self.grand2_layer, self.sh_gcn, self.e_s0,
                                        (self.adj_sh.T * self.sh_weight.T)[self.grand_herb[1]])
            e_h_grand2_norm = self.neighbor_aggregation(e_h_grand2)

            # 梯度3邻居信息聚合
            e_s_grand3 = self.gat_layer(self.grand3_layer, self.sh_gcn, self.e_h0,
                                        (self.adj_sh * self.sh_weight)[self.grand_sym[2]])
            e_s_grand3_norm = self.neighbor_aggregation(e_s_grand3)
            e_h_grand3 = self.gat_layer(self.grand3_layer, self.sh_gcn, self.e_s0,
                                        (self.adj_sh.T * self.sh_weight.T)[self.grand_herb[2]])
            e_h_grand3_norm = self.neighbor_aggregation(e_h_grand3)

            # 合并矩阵
            e_s = self.conform_grade_matrix(e_s_grand1_norm, e_s_grand2_norm, e_s_grand3_norm, self.num_s)
            e_h = self.conform_grade_matrix(e_h_grand1_norm, e_h_grand2_norm, e_h_grand3_norm, self.num_h)

            # SVD增强视图SH图
            # 梯度1邻居信息聚合
            z_s_grand1 = self.gat_layer(self.grand1_layer, self.sh_gcn, self.e_h0,
                                        (self.svd_s * self.sh_weight)[self.grand_sym[0]])
            z_s_grand1_norm = self.neighbor_aggregation(z_s_grand1)
            z_h_grand1 = self.gat_layer(self.grand1_layer, self.sh_gcn, self.e_s0,
                                        (self.svd_h * self.sh_weight.T)[self.grand_herb[0]])
            z_h_grand1_norm = self.neighbor_aggregation(z_h_grand1)

            # 梯度2邻居信息聚合、
            z_s_grand2 = self.gat_layer(self.grand2_layer, self.sh_gcn, self.e_h0,
                                        (self.svd_s * self.sh_weight)[self.grand_sym[1]])
            z_s_grand2_norm = self.neighbor_aggregation(z_s_grand2)
            z_h_grand2 = self.gat_layer(self.grand2_layer, self.sh_gcn, self.e_s0,
                                        (self.svd_h * self.sh_weight.T)[self.grand_herb[1]])
            z_h_grand2_norm = self.neighbor_aggregation(z_h_grand2)

            # 梯度3邻居信息聚合
            z_s_grand3 = self.gat_layer(self.grand3_layer, self.sh_gcn, self.e_h0,
                                        (self.svd_s * self.sh_weight)[self.grand_sym[2]])
            z_s_grand3_norm = self.neighbor_aggregation(z_s_grand3)
            z_h_grand3 = self.gat_layer(self.grand3_layer, self.sh_gcn, self.e_s0,
                                        (self.svd_h * self.sh_weight.T)[self.grand_herb[2]])
            z_h_grand3_norm = self.neighbor_aggregation(z_h_grand3)

            # 合并矩阵
            z_s = self.conform_grade_matrix(z_s_grand1_norm, z_s_grand2_norm, z_s_grand3_norm, self.num_s)
            z_h = self.conform_grade_matrix(z_h_grand1_norm, z_h_grand2_norm, z_h_grand3_norm, self.num_h)

            # 症状同构图
            # ss_weighth和hh_weighth构造
            weights_ss, weights_hh = self.construct_ss_hh_weight()
            e_ss_grand1 = self.gat_layer(self.grand1_layer, self.ss_gcn, self.e_ss0,
                                         (self.adj_ss * weights_ss)[self.grand_sym[0]])
            e_ss_grand1_norm = self.neighbor_aggregation(e_ss_grand1)
            e_ss_grand2 = self.gat_layer(self.grand2_layer, self.ss_gcn, self.e_ss0,
                                         (self.adj_ss * weights_ss)[self.grand_sym[1]])
            e_ss_grand2_norm = self.neighbor_aggregation(e_ss_grand2)
            e_ss_grand3 = self.gat_layer(self.grand3_layer, self.ss_gcn, self.e_ss0,
                                         (self.adj_ss * weights_ss)[self.grand_sym[2]])
            e_ss_grand3_norm = self.neighbor_aggregation(e_ss_grand3)
            # 合并矩阵
            e_ss = self.conform_grade_matrix(e_ss_grand1_norm, e_ss_grand2_norm, e_ss_grand3_norm, self.num_s)

            # 中药同构图
            e_hh_grand1 = self.gat_layer(self.grand1_layer, self.hh_gcn, self.e_hh0,
                                         (self.adj_hh * weights_hh)[self.grand_herb[0]])
            e_hh_grand1_norm = self.neighbor_aggregation(e_hh_grand1)
            e_hh_grand2 = self.gat_layer(self.grand2_layer, self.hh_gcn, self.e_hh0,
                                         (self.adj_hh * weights_hh)[self.grand_herb[1]])
            e_hh_grand2_norm = self.neighbor_aggregation(e_hh_grand2)
            e_hh_grand3 = self.gat_layer(self.grand3_layer, self.hh_gcn, self.e_hh0,
                                         (self.adj_hh * weights_hh)[self.grand_herb[2]])
            e_hh_grand3_norm = self.neighbor_aggregation(e_hh_grand3)
            # 合并矩阵
            e_hh = self.conform_grade_matrix(e_hh_grand1_norm, e_hh_grand2_norm, e_hh_grand3_norm, self.num_h)
        '''


        '''
            # 1.SH主视图及其邻居信息聚合和消息传递
            # 1.1 用创建的函数对节点进行随机初始化表示
            # 1.2 一阶邻居信息聚合
            e_s1 = self.sh_gcn(self.e_h0, self.adj_sh * self.sh_weight)  # (360,753)@(753,64)=(360,64)
            e_h1 = self.sh_gcn(self.e_s0, self.adj_sh.T * self.sh_weight.T)  # (753,360)@(360,64)=(360,64)

            

            # 1.3 二阶邻居信息聚合
            e_s2 = self.sh_gcn(e_h1, self.adj_sh * self.sh_weight)
            e_h2 = self.sh_gcn(e_s1, self.adj_sh.T * self.sh_weight.T)

            # 1.4 将三层信息进行融合，得到更新后目标节点的信息，对一阶邻居、二阶邻居信息现进行归一化再聚合起来
            e_s = (self.e_s0 + e_s1 / (torch.norm(e_s1, p=2, dim=1).view(-1, 1)) + e_s2 / (
                torch.norm(e_s2, p=2, dim=1).view(-1, 1))) / 3.0
            e_h = (self.e_h0 + e_h1 / (torch.norm(e_h1, p=2, dim=1).view(-1, 1)) + e_h2 / (
                torch.norm(e_h2, p=2, dim=1).view(-1, 1))) / 3.0

            # 2.svd重构的邻接图SH进行邻居信息聚合和消息传递
            # 2.1 共享节点初始化表示

            z_s0 = self.e_s0
            z_h0 = self.e_h0

            # 和原框架不同，我这里的邻居信息聚合加入了可学习权重，由于可学习权重是随机的，所以聚合而成的邻居信息与原图中是不一样的，原框架是直接共享了原图的邻居信息
            # 2.2 一阶邻居信息聚合
            z_s1 = self.sh_gcn(self.e_h0, self.svd_s)
            z_h1 = self.sh_gcn(self.e_s0, self.svd_h)

            # 2.3 二阶邻居信息聚合
            z_s2 = self.sh_gcn(e_h1, self.svd_s)
            z_h2 = self.sh_gcn(e_s1, self.svd_h)

            # 2.4 将三层信息进行融合，得到更新后目标节点的信息，对一阶邻居、二阶邻居信息现进行归一化再聚合起来
            z_s = (z_s0 + z_s1 / (torch.norm(z_s1, p=2, dim=1).view(-1, 1)) + z_s2 / (
                torch.norm(z_s2, p=2, dim=1)).view(-1, 1)) / 3.0
            z_h = (z_h0 + z_h1 / (torch.norm(z_h1, p=2, dim=1).view(-1, 1)) + z_h2 / (
                torch.norm(z_h2, p=2, dim=1)).view(-1, 1)) / 3.0

            # 3.SS/HH主视图及其邻居信息聚合和消息传递
            # 3.1 用创建的函数对节点进行随机初始化表示

            # 3.2 一阶邻居信息聚合

            # ss_weighth和hh_weighth构造
            weights_ss, weights_hh = self.construct_ss_hh_weight()

            e_ss1 = self.sh_gcn(self.e_ss0, self.adj_ss * weights_ss)
            e_hh1 = self.sh_gcn(self.e_hh0, self.adj_hh * weights_hh)

            # 3.3 二阶邻居信息聚合
            e_ss2 = self.sh_gcn(e_ss1, self.adj_ss * weights_ss)
            e_hh2 = self.sh_gcn(e_hh1, self.adj_hh * weights_hh)

            # 3.4 将三层信息进行融合，得到更新后目标节点的信息，对一阶邻居、二阶邻居信息现进行归一化再聚合起来
            e_ss = (self.e_ss0 + e_ss1 / (torch.norm(e_ss1, p=2, dim=1).view(-1, 1)) + e_ss2 / (
                torch.norm(e_ss2, p=2, dim=1).view(-1, 1))) / 3.0
            e_hh = (self.e_hh0 + e_hh1 / (torch.norm(e_hh1, p=2, dim=1).view(-1, 1)) + e_hh2 / (
                torch.norm(e_hh2, p=2, dim=1).view(-1, 1))) / 3.0

            # 4.从主视图SH、SS和HH症状s和中药h的信息聚合成最终的节点表示
            self.g_s = (e_s / (torch.norm(e_s, p=2, dim=1).view(-1, 1)) + e_ss / (
                torch.norm(e_ss, p=2, dim=1).view(-1, 1))) / 2.0
            self.g_h = (e_h / (torch.norm(e_h, p=2, dim=1).view(-1, 1)) + e_hh / (
                torch.norm(e_hh, p=2, dim=1).view(-1, 1))) / 2.0
            

            # 5.由症状节点和中药节点和处方训练集得出处方对中药的预测
            pres_s = torch.mm(prescription, self.g_s)
            sum_pre = prescription.sum(dim=1).view(-1, 1)
            pres_s = pres_s / sum_pre
            pres_s = self.SI_bn(pres_s)
            pres_s = self.relu(pres_s)
            pres_h = torch.mm(pres_s, self.g_h.T)

            # 6.计算对比损失cl,这里计算相似度用点积进行计算 ***可以到时候进行替换相似度算法,如
            pos_score = torch.clamp((z_s[sids] * e_s[sids]).sum(1) / self.temp, -5.0, 5.0).mean() + torch.clamp(
                (z_h[hids] * e_h[hids]).sum(1) / self.temp, -5.0, 5.0).mean()
            neg_score = torch.log((torch.exp(z_s[sids] @ e_s.T) / self.temp).sum(1) + 1e-8).mean() + torch.log((torch.exp(z_h[hids] @ e_h.T) / self.temp).sum(1) + 1e-8).mean()
            loss_cl = -pos_score + neg_score

            # 7.bpr loss
            s_emb = self.g_s[sids]
            pos_emb = self.g_h[pos]
            neg_emb = self.g_h[neg]
            pos_scores = (s_emb * pos_emb).sum(-1)
            neg_scores = (s_emb * neg_emb).sum(-1)
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # 8.reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            loss = self.lambda_1 * loss_cl + loss_reg
            return loss, loss_r, self.lambda_1 * loss_cl, pres_h
        '''
