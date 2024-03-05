import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import random

import networkx as nx
from torch.utils.data.dataset import T_co
from sklearn.cluster import KMeans

seed = 2023
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)


#从s-h图中得到s-s的相似性以及h-h的相似性
def getSim_from_sh(adj_oh):
    num_s = adj_oh.shape[0]
    num_h = adj_oh.shape[1]
    #得到s-s的相似性
    # 计算第一行与第二行相等元素值的个数
    ss_sim = np.zeros((num_s,num_s))
    for s1 in range(num_s):
        for s2 in range(s1,num_s):
            equal_count = np.sum(adj_oh[s1] == adj_oh[s2])
            ss_sim[s1][s2]=equal_count/num_h
            ss_sim[s2][s1]=equal_count/num_h

    #得到h-h的相似性
    hh_sim = np.zeros((num_h, num_h))
    for h1 in range(num_h):
        for h2 in range(h1, num_h):
            equal_count = np.sum(adj_oh[:,h1] == adj_oh[:,h2])
            hh_sim[h1][h2] = equal_count / num_s
            hh_sim[h2][h1] = equal_count / num_s

    return ss_sim,hh_sim

# normalizing the adj matrix
def normal_adj_matrix(coo_adj):
    rowD = np.array(coo_adj.sum(1)).squeeze()
    colD = np.array(coo_adj.sum(0)).squeeze()
    for i in range(len(coo_adj.data)):
        coo_adj.data[i] = coo_adj.data[i] / pow(rowD[coo_adj.row[i]] * colD[coo_adj.col[i]], 0.5)

    #当有无穷大的值时，将该元素值置为0
    coo_adj.data[np.isinf(coo_adj.data)] = 0.
    return coo_adj


def get_correlation_coefficient(adj_oh):
    neighbors = {i:[] for i in range(len(adj_oh))}

    row,col=  np.where(adj_oh!=0)

    start = 0
    k=0
    for i,r in enumerate(row):
        end = r
        if start!=end:
            neighbors[start]=col[k:i].tolist()
            start = end
            k=i
    return neighbors

def jaccard_similarity(neighbor1, neighbor2):
    intersection = len(set(neighbor1).intersection(neighbor2))
    union = len(set(neighbor1).union(neighbor2))
    if union==0:
        return 0
    return intersection / union

def get_similarity(neighbors):
    # 创建一个空的相似性矩阵，用于存储节点之间的 Jaccard 相关性
    num_nodes = len(neighbors)
    similarity_matrix = np.zeros((num_nodes, num_nodes))

    # 计算每一对节点之间的 Jaccard 相关性
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            node1 = i
            node2 = j
            neighbor1 = neighbors[node1]
            neighbor2 = neighbors[node2]
            similarity = jaccard_similarity(neighbor1, neighbor2)
            similarity_matrix[node1, node2] = similarity
            similarity_matrix[node2, node1] = similarity

    # 打印节点之间的 Jaccard 相关性矩阵
    return similarity_matrix

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row,sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices,values,shape)

def sparse_dropout(mat,dropout):
    if dropout==0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(),p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices,values,size)

class TrnData(Dataset):
    def __init__(self,coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.float32)

    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(self.dokmat.shape[1])
                if (u,i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg
        np.random.shuffle(self.negs)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx],self.cols[idx],self.negs[idx]



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


#通过元路径构造ss图和hh图，先把元路径构造出来，然后按顺序排列

def construct_ss_graph(sh_graph):
    # 获取所有元路径
    meta_paths = []
    # 遍历每个症状
    for s1 in range(sh_graph.shape[0]):
        # 找到s1连接的所有中药
        connected_herbs = np.where(sh_graph[s1] == 1)[0]
        for h in connected_herbs:
            # 找到与中药h连接的所有其他症状
            connected_symptoms = np.where(sh_graph[:, h] == 1)[0]
            for s2 in connected_symptoms:
                if (s1 != s2):  # 确保s1和s2不是同一个症状
                # if (s1 != s2) and ((s2, h, s1) not in meta_paths):  # 确保s1和s2不是同一个症状
                    meta_paths.append((s1, h, s2))
                    # print(len(meta_paths))
    #对所有元路径进行排序
    # 按照第0个元素，第2个元素，第1个元素的顺序对列表进行排序
    sorted_triplets = sorted(meta_paths, key=lambda x: (x[0], x[2], x[1]))
    return sorted_triplets

def construct_hh_graph(hs_graph):
    # 获取所有元路径
    meta_paths = []
    # 遍历每个症状
    for h1 in range(hs_graph.shape[0]):
        # 找到s1连接的所有中药
        connected_herbs = np.where(hs_graph[h1] == 1)[0]
        for s in connected_herbs:
            # 找到与中药h连接的所有其他症状
            connected_symptoms = np.where(hs_graph[:, s] == 1)[0]
            for h2 in connected_symptoms:
                if (h1 != h2):  # 确保s1和s2不是同一个症状
                # if (h1 != h2) and ((h2, s, h1) not in meta_paths):  # 确保s1和s2不是同一个症状
                    meta_paths.append((h1, s, h2))
                    # print(len(meta_paths))

    # 对所有元路径进行排序
    # 按照第0个元素，第2个元素，第1个元素的顺序对列表进行排序
    sorted_triplets = sorted(meta_paths, key=lambda x: (x[0], x[2], x[1]))
    return sorted_triplets

#分梯度
def get_node_grand(fre_herb,fre_sym,num_train,method_s=0,method_h=0):
    '''
    method=0时：
    第一梯度：节点频数/训练集处方数>=10%
    第二梯度：30%<=节点频数/训练集处方数<10%
    第三梯度：节点频数/训练集处方数<30%

    method=1时：
    第一梯度：节点频数/节点数>=10%
    第二梯度：30%<=节点频数/节点数<10%
    第三梯度：节点频数/节点数<30%

    method=2时：
    试试StartMed论文中的分层方法
    '''
    #中药频数分层
    grand_h=[[],[],[]]

    if method_h==0:
        q1=int(num_train*0.1)
        q2=int(num_train*0.05)

        for herb in fre_herb:
            if herb[1]/num_train>=0.1:
                grand_h[0].append(herb[0])
            elif q1>herb[1]>=q2:
                grand_h[1].append(herb[0])
            else:
                grand_h[2].append(herb[0])
    elif method_h==1:
        num_h=len(fre_herb)
        sorted_data = sorted(fre_herb, key=lambda x: x[1], reverse=True)
        sorted_data = [[int(item[0]), int(item[1])] for item in sorted_data]
        sorted_grand_h=[[],[],[]]
        # 计算各梯度的分界点
        first_tier_threshold = int(num_h * 0.10)
        second_tier_threshold = int(num_h * 0.30)
        grand_h[0].extend([item[0] for item in sorted_data[:first_tier_threshold]])
        sorted_grand_h[0]=sorted(grand_h[0])
        grand_h[1].extend([item[0] for item in sorted_data[first_tier_threshold:second_tier_threshold]])
        sorted_grand_h[1] = sorted(grand_h[1])
        grand_h[2].extend([item[0] for item in sorted_data[second_tier_threshold:]])
        sorted_grand_h[2] = sorted(grand_h[2])
        grand_h=sorted_grand_h
    else:
        #KMeans分类
        node_frequencies = fre_herb[:, 1]
        X = np.array(node_frequencies).reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
        labels = kmeans.labels_
        # 如果需要，您可以查看每个聚类的中心值
        cluster_centers = kmeans.cluster_centers_
        print("三层的聚类中心为：",cluster_centers)
        for i, label in enumerate(labels):
            # print(f"节点{i}属于层{label}")
            if label==0:
                grand_h[2].append(i)
            elif label==1:
                grand_h[0].append(i)
            else:
                grand_h[1].append(i)


    # 症状频数分层
    grand_s = [[], [], []]
    if method_s==0:
        q1=int(num_train*0.1)
        q2=int(num_train*0.05)

        for sym in fre_sym:
            if sym[1]>=q1:
                grand_s[0].append(sym[0])
            elif q1>sym[1]>=q2:
                grand_s[1].append(sym[0])
            else:
                grand_s[2].append(sym[0])
    elif method_s==1:
        num_s=len(fre_sym)
        sorted_data = sorted(fre_sym, key=lambda x: x[1], reverse=True)
        sorted_data = [[int(item[0]), int(item[1])] for item in sorted_data]
        sorted_grand_s = [[], [], []]
        # 计算各梯度的分界点
        first_tier_threshold = int(num_s * 0.10)
        second_tier_threshold = int(num_s * 0.30)
        grand_s[0].extend([item[0] for item in sorted_data[:first_tier_threshold]])
        sorted_grand_s[0] = sorted(grand_s[0])
        grand_s[1].extend([item[0] for item in sorted_data[first_tier_threshold:second_tier_threshold]])
        sorted_grand_s[1] = sorted(grand_s[1])
        grand_s[2].extend([item[0] for item in sorted_data[second_tier_threshold:]])
        sorted_grand_s[2] = sorted(grand_s[2])
        grand_s = sorted_grand_s
    else:
        node_frequencies = fre_sym[:, 1]
        X = np.array(node_frequencies).reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        print("三层的聚类中心为：", cluster_centers)
        for i, label in enumerate(labels):
            # print(f"节点{i}属于层{label}")
            if label == 0:
                grand_s[2].append(i)
            elif label == 1:
                grand_s[0].append(i)
            else:
                grand_s[1].append(i)

    return grand_h,grand_s



def construct_ss_hh_weight(num_s,num_h,ss_metapath,hh_metapath,sh_weight,hs_weight):

    # 计算症状同构图
    symptom_graph_weights = np.zeros((num_s, num_s))
    k=0
    for i,(s1, h, s2) in enumerate(ss_metapath):
        k+=1
        weight = (sh_weight[s1, h]+ sh_weight[s2, h]) / 2
        symptom_graph_weights[s1, s2] += weight
        symptom_graph_weights[s2, s1] += weight
        if (i+1)==len(ss_metapath):
            symptom_graph_weights[s1, s2] /= k
            symptom_graph_weights[s2, s1] /= k
            break
        if ss_metapath[i+1][0]!=ss_metapath[i][0] or ss_metapath[i+1][2]!=ss_metapath[i][2]:
            symptom_graph_weights[s1, s2]/=k
            symptom_graph_weights[s2, s1]/=k
            k=0


    # 将对角线置为0
    # torch.fill_diagonal_(symptom_graph_weights, 0)

    # 计算中药同构图
    herb_graph_weights = np.zeros((num_h, num_h))
    q=0
    for j,(h1, s, h2) in enumerate(hh_metapath):
        q+=1
        weight = (hs_weight[h1, s]+ hs_weight[h2, s]) / 2
        herb_graph_weights[h1, h2] += weight
        herb_graph_weights[h2, h1] += weight
        if (j+1)==len(hh_metapath):
            herb_graph_weights[h1, h2] /= q
            herb_graph_weights[h2, h1] /= q
            break
        if hh_metapath[j+1][0]!=hh_metapath[j][0] or hh_metapath[j+1][2]!=hh_metapath[j][2]:
            herb_graph_weights[h1, h2]/=q
            herb_graph_weights[h2, h1]/=q
            q=0
    return symptom_graph_weights, herb_graph_weights




