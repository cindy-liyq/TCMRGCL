import numpy as np
from numpy.linalg import solve
import time
from collections import defaultdict
import torch
#用ppr计算节点之间的相关度
def ppr(num_s,num_h,adj):
    alpha = 0.8
    n=num_s+num_h
    vertex = np.arange(n)
    full_adj = np.zeros((n,n))
    full_adj[:num_s,:num_h]=adj
    full_adj[num_s:, num_h:]=adj.T
    M=full_adj/full_adj.sum(1)
    A = np.eye(n) - alpha * M.T
    begin = time.time()
    D = np.linalg.inv(A)
    end = time.time()
    print('用时：', end - begin)
    ppr_score = defaultdict(lambda: [[] for _ in range(2)])
    for j in range(n):
        score = {}
        total = 0.0  # 用于归一化
        for i in range(n):
            score[vertex[i]] = D[i, j]
            total += D[i, j]
        li = sorted(score.items(), key=lambda x: x[1], reverse=True)

        for ele in li:
            if j<num_s:#对于症状，保留中药
                if ele[0]>(num_s-1):
                    ppr_score[j][0].append(ele[0])#保存节点id
                    ppr_score[j][1].append(ele[1] / total)#保存相关度的值
            else:#对于中药，保留症状
                if ele[0]<num_s:
                    ppr_score[j][0].append(ele[0])
                    ppr_score[j][1].append(ele[1] / total)

    #将PPR_score隐射为一个矩阵，表示异构图所有节点之间的相关度
    PPR_sh = np.zeros((num_s,num_h))
    PPR_hs = np.zeros((num_h,num_s))
    for id,value in ppr_score.items():
        if id<num_s:
            PPR_sh[id][np.array(value[0])-num_s]=value[1]
        else:
            PPR_hs[id-360][np.array(value[0])] = value[1]
    PPR_sh = PPR_sh+PPR_hs.T
    return PPR_sh
