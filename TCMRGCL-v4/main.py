import numpy as np
import torch
import pickle
import pandas as pd
from parser import args
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from utils import normal_adj_matrix, TrnData, scipy_sparse_mat_to_torch_sparse_tensor, get_correlation_coefficient, \
    get_similarity, getSim_from_sh, EarlyStopping, get_node_grand

import networkx as nx
from PPR import ppr
from torch.utils.tensorboard import SummaryWriter
import torch.autograd

seed = 2023
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

# hyperparameters
dim = args.hidden_dim  # 256
layer = args.gnn_layer  # 2
temp = args.temp  # 0.2
batch_user = args.batch  # 128
epoch_no = args.epoch  # 100
max_samp = 40  # 40
lambda_1 = args.lambda1  # 0.2
lambda_2 = args.lambda2  # 1e-07
lambda_3 = args.lambda3  # 1e-07
dropout = args.dropout  # 0.0
lr = args.lr  # 0.001
decay = args.decay  # 0.99
svd_q = args.q  # 当 svd_q 等于 5 时，表示在低秩奇异值分解（SVD）中，只保留矩阵的前 5 个最大奇异值及其对应的奇异向量
path = args.path
batch_size = args.inter_batch  # 4096
device = args.device
loop = args.loop
method_s = args.method_s
method_h = args.method_h
grand1_layer,grand2_layer,grand3_layer=args.grand1_layer,args.grand2_layer,args.grand3_layer
dev = args.dev_dataset

# load edge data------SMGCN
## s-h
sym_herb_path = path + 'sh_train_onehot.csv'
sh_onehot = pd.read_csv(sym_herb_path, dtype=np.float32).values[:, 1:]
coo_sh = coo_matrix(sh_onehot)
train_csr = (coo_sh != 0).astype(np.float32)
## s-s
sym_sym_path = path + 'symPair_onehot.csv'
ss_onehot = pd.read_csv(sym_sym_path, dtype=np.float32).values[:, 1:]
coo_ss = coo_matrix(ss_onehot)
## h-h
herb_herb_path = path + 'herbPair_onehot.csv'
hh_onehot = pd.read_csv(herb_herb_path, dtype=np.float32).values[:, 1:]
coo_hh = coo_matrix(hh_onehot)
print('Data loaded.')

'''
# 用PPR计算sh图中症状节点和中药节点之间的相关度作为边的权重-------------
ppr_sh_array = ppr(sh_onehot.shape[0], sh_onehot.shape[1], sh_onehot)
# 边的权重
sh_weight = ppr_sh_array * sh_onehot

# 构造元路径得到症状同构图和中药同构图
print("症状元路径开始时间：",time.time())
ss_metapath = construct_ss_graph(sh_onehot)
print("中药元路径开始时间：",time.time())
hh_metapath = construct_hh_graph(sh_onehot.T)
symptom_graph_weights, herb_graph_weights=construct_ss_hh_weight(sh_onehot.shape[0], sh_onehot.shape[1],ss_metapath,hh_metapath,sh_weight,sh_weight.T)
sh_weight = torch.from_numpy(sh_weight).to(device)
ss_weight = torch.from_numpy(symptom_graph_weights).to(device)
hh_weight = torch.from_numpy(herb_graph_weights).to(device)
'''

if 'Set2Set' in path:

    # 加验证集
    if dev==1:
        # 导入处方数据------SMGCN
        pres_sym_tapath = path + 'ps_train_onehot.csv'
        train_data = pd.read_csv(pres_sym_tapath, dtype=np.float32).values
        # 把训练集拆分成80%训练集和20%测试集
        p_list = [x for x in range(len(train_data))]
        x_train, x_dev = train_test_split(p_list, test_size=0.1, shuffle=False,
                                          random_state=2023)
        ps_train_onehot = pd.read_csv(pres_sym_tapath, dtype=np.float32).values[:x_train[-1] + 1, 1:]
        pres_sym_tepath = path + 'ps_test_onehot.csv'
        ps_test_onehot = pd.read_csv(pres_sym_tepath, dtype=np.float32).values[:, 1:]
        pres_herb_tapath = path + 'ph_train_onehot.csv'
        ph_train_onehot = pd.read_csv(pres_herb_tapath, dtype=np.float32).values[:x_train[-1] + 1, 1:]
        pres_herb_tepath = path + 'ph_test_onehot.csv'
        ph_test_onehot = pd.read_csv(pres_herb_tepath, dtype=np.float32).values[:, 1:]

        ps_dev_onehot = pd.read_csv(pres_sym_tapath, dtype=np.float32).values[x_dev[0]:, 1:]
        ph_dev_onehot = pd.read_csv(pres_herb_tapath, dtype=np.float32).values[x_dev[0]:, 1:]

        num_Ptrain = len(ps_train_onehot)
        num_Pdev = len(ps_dev_onehot)
        num_Ptest = len(ps_test_onehot)
        print(f"训练集的数量为:{num_Ptrain},验证集的数量为：{num_Pdev},测试集的数量为:{num_Ptest}")
    else:
    # 不加验证集
        print("使用的数据集是SMGCN")
        # 导入处方数据------SMGCN
        pres_sym_tapath = path + 'ps_train_onehot.csv'
        ps_train_onehot = pd.read_csv(pres_sym_tapath, dtype=np.float32).values[:, 1:]
        pres_sym_tepath = path + 'ps_test_onehot.csv'
        ps_test_onehot = pd.read_csv(pres_sym_tepath, dtype=np.float32).values[:, 1:]
        pres_herb_tapath = path + 'ph_train_onehot.csv'
        ph_train_onehot = pd.read_csv(pres_herb_tapath, dtype=np.float32).values[:, 1:]
        pres_herb_tepath = path + 'ph_test_onehot.csv'
        ph_test_onehot = pd.read_csv(pres_herb_tepath, dtype=np.float32).values[:, 1:]

        # 导入症状、中药频数数据
        frequency_herb_path = path + 'idx_fre_herb.csv'
        frequency_herb = pd.read_csv(frequency_herb_path, dtype=np.float32).values[:, 1:3]
        frequency_sym_path = path + 'idx_fre_sym.csv'
        frequency_sym = pd.read_csv(frequency_sym_path, dtype=np.float32).values[:, 1:3]
        sh_weight_path = path + 'as_sh.csv'
        sh_weight = pd.read_csv(sh_weight_path, dtype=np.float32).values[:, 1:]
        sh_weight = torch.from_numpy(sh_weight).to(device)
    num_Ptrain = len(ps_train_onehot)
    num_Ptest = len(ps_test_onehot)
    print(f"训练集的数量为:{num_Ptrain},测试集的数量为:{num_Ptest}")

else:
    # 导入处方数据------KDHR
    if dev == 1:
    # 加验证集
        pres_sym_path = path + 'pS_onehot_total.csv'
        ps_train_onehot = pd.read_csv(pres_sym_path, dtype=np.float32).values[:20259, 1:]
        ps_dev_onehot = pd.read_csv(pres_sym_path, dtype=np.float32).values[20259:27012, 1:]
        ps_test_onehot = pd.read_csv(pres_sym_path, dtype=np.float32).values[27012:, 1:]
        pres_herb_path = path + 'pH_onehot_total.csv'
        ph_train_onehot = pd.read_csv(pres_herb_path, dtype=np.float32).values[:20259, 1:]
        ph_dev_onehot = pd.read_csv(pres_herb_path, dtype=np.float32).values[20259:27012, 1:]
        ph_test_onehot = pd.read_csv(pres_herb_path, dtype=np.float32).values[27012:, 1:]
        num_Ptrain = len(ps_train_onehot)
        num_Ptest = len(ps_test_onehot)
        num_Pdev = len(ps_dev_onehot)
        print(f"训练集的数量为:{num_Ptrain},验证集的数量为：{num_Pdev},测试集的数量为:{num_Ptest}")

    else:
        # 不加验证集
        pres_sym_path = path + 'pS_onehot_total.csv'
        ps_train_onehot = pd.read_csv(pres_sym_path, dtype=np.float32).values[:20259, 1:]
        ps_test_onehot = pd.read_csv(pres_sym_path, dtype=np.float32).values[20259:, 1:]
        pres_herb_path = path + 'pH_onehot_total.csv'
        ph_train_onehot = pd.read_csv(pres_herb_path, dtype=np.float32).values[:20259, 1:]
        ph_test_onehot = pd.read_csv(pres_herb_path, dtype=np.float32).values[20259:, 1:]

        # 导入症状、中药频数数据
        frequency_herb_path = path + 'idx_fre_herb.csv'
        frequency_herb = pd.read_csv(frequency_herb_path, dtype=np.float32).values[:, 1:3]
        frequency_sym_path = path + 'idx_fre_sym.csv'
        frequency_sym = pd.read_csv(frequency_sym_path, dtype=np.float32).values[:, 1:3]
        sh_weight_path = path + 'as_sh.csv'
        sh_weight = pd.read_csv(sh_weight_path, dtype=np.float32).values[:, 1:]
        sh_weight = torch.from_numpy(sh_weight).to(device)
    num_Ptrain = len(ps_train_onehot)
    num_Ptest = len(ps_test_onehot)
    print(f"训练集的数量为:{num_Ptrain},测试集的数量为:{num_Ptest}")


if __name__ == '__main__':

    # writer = SummaryWriter(log_dir="runs/writer_results")
    print("开始运行：",time.time())
    grand_herb, grand_sym = get_node_grand(frequency_herb, frequency_sym, num_Ptrain, method_s, method_h)
    precision = []
    recall = []
    f1score = []
    while loop:
        loop -= 1

        # normalizing the adj matrix
        ##s-h
        normal_sh = normal_adj_matrix(coo_sh)
        ##s-s
        normal_ss = normal_adj_matrix(coo_ss)
        ##h-h
        normal_hh = normal_adj_matrix(coo_hh)

        # construct data loader
        train_dataset = TrnData(normal_sh)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        # 加验证集
        '''
        dev_dataset = TrnData(normal_sh)  # 1111111111111111
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)  # 1111111111111111
        '''

        test_dataset = TrnData(normal_sh)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        adj_norm_sh = scipy_sparse_mat_to_torch_sparse_tensor(normal_sh).coalesce().to(device)
        adj_norm_ss = scipy_sparse_mat_to_torch_sparse_tensor(normal_ss).coalesce().to(device)
        adj_norm_hh = scipy_sparse_mat_to_torch_sparse_tensor(normal_hh).coalesce().to(device)
        print('Adj matrix normalized.')


        origin_data = {}
        origin_data['train_loader']=train_loader
        origin_data['test_loader']=test_loader
        origin_data['num_Ptrain']=num_Ptrain
        origin_data['num_Ptest']=num_Ptest
        origin_data['ps_train_onehot']=ps_train_onehot
        origin_data['ph_train_onehot']=ph_train_onehot
        origin_data['ps_test_onehot']=ps_test_onehot
        origin_data['ph_test_onehot']=ph_test_onehot

        #############################添加相似度

        # 从s-h中获取ss相似性和hh相似性+++++++++++++++++++++++++++

        sim_ss, sim_hh = getSim_from_sh(sh_onehot)
        sim_ss_data = pd.DataFrame(sim_ss)
        sim_hh_data = pd.DataFrame(sim_hh)
        sim_ss_data.to_csv("sim_ss.csv")
        sim_hh_data.to_csv("sim_hh.csv")
        print("相似性计算已完成")
        sim_ss = torch.FloatTensor(sim_ss).to(device)
        sim_hh = torch.FloatTensor(sim_hh).to(device)
        # 计算s-s和h-h中的jaccard相似性，用于表示症状-症状(中药-中药)共同出现的概率，用于衡量症状(中药)之间的相互作用

        ss_neighbors = get_correlation_coefficient(ss_onehot)
        jacc_ss = torch.FloatTensor(get_similarity(ss_neighbors)).to(device)
        jacc_ss = 0.5*sim_ss+0.5*jacc_ss  # +++++++++++++++++++++++++++++++++++

        hh_neighbors = get_correlation_coefficient(hh_onehot)
        jacc_hh = torch.FloatTensor(get_similarity(hh_neighbors)).to(device)
        jacc_hh = 0.5*sim_hh+0.5*jacc_hh  # +++++++++++++++++++++++++++++++++++

        adj_norm_ss = adj_norm_ss.to_dense() * jacc_ss  # *是对应元素相乘
        adj_norm_hh = adj_norm_hh.to_dense() * jacc_hh
        adj_norm_sh = adj_norm_sh.to_dense() * sh_weight
        #############################添加相似度


        loss_list = []
        loss_r_list = []
        loss_s_list = []
        loss_pre_list = []

        from model_noLSNL import FRGCL

        model = FRGCL(adj_norm_sh.shape[0], adj_norm_sh.shape[1], adj_norm_sh, adj_norm_ss,
                      adj_norm_hh, dim, layer, temp, lambda_1, lambda_2, dropout, batch_user, device, grand_herb,
                      grand_sym, grand1_layer, grand2_layer, grand3_layer,epoch_no,origin_data).to(device)
        #
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr=lr)
        # criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        # early_stopping = EarlyStopping(patience=7, verbose=True)
        print("------------------------------------------train start----------------------------------------------------------")


        print("开始训练：", time.time())
        model.train()

    #     for epoch in range(epoch_no):
    #         # if (epoch + 1) % 50 == 0:
    #         #     torch.save(model.state_dict(), 'saved_model/saved_model_epoch_' + str(epoch) + '.pt')
    #         #     torch.save(optimizer.state_dict(), 'saved_model/saved_optim_epoch_' + str(epoch) + '.pt')
    #
    #         epoch_loss = 0
    #         # epoch_loss_r = 0
    #         epoch_loss_s = 0
    #         epoch_loss_pre = 0
    #         last_loss_pre = 0
    #         train_loader.dataset.neg_sampling()
    #         batch_num = len(train_loader)
    #         batch_no = int(np.ceil(num_Ptrain / batch_num))
    #         for i, batch in enumerate(tqdm(train_loader)):
    #             sids, pos, neg = batch
    #             sids = sids.long().to(device)
    #             pos = pos.long().to(device)
    #             neg = neg.long().to(device)
    #
    #             start = i * batch_no
    #             end = min((i + 1) * batch_no, num_Ptrain)
    #             ps = ps_train_onehot[start:end]
    #             ph = ph_train_onehot[start:end]
    #             ps = torch.tensor(ps).to(device)
    #             ph = torch.tensor(ph).to(device)
    #
    #             hids = torch.concat([pos, neg], dim=0)
    #
    #             optimizer.zero_grad()
    #             # loss, loss_r, loss_s, prediction_result = model(sids, hids, pos, neg, ps, test=False)
    #             loss, loss_s, prediction_result = model(sids, hids, pos, neg, ps, test=False)
    #             loss_pre = criterion(prediction_result, ph)
    #             loss_total = loss + lambda_3 * loss_pre
    #             loss_total.backward()
    #             optimizer.step()
    #
    #             epoch_loss += loss_total.cpu().item()
    #             # epoch_loss_r += loss_r.cpu().item()
    #             epoch_loss_s += loss_s.cpu().item()
    #             epoch_loss_pre += loss_pre.cpu().item()
    #         batch_no = len(train_loader)
    #         epoch_loss = epoch_loss / batch_no
    #         # epoch_loss_r = epoch_loss_r / batch_no
    #         epoch_loss_s = epoch_loss_s / batch_no
    #         epoch_loss_pre = epoch_loss_pre / batch_no
    #         loss_list.append(epoch_loss)
    #         # loss_r_list.append(epoch_loss_r)
    #         loss_s_list.append(epoch_loss_s)
    #         loss_pre_list.append(epoch_loss_pre)
    #         tags=['train_loss','epoch_loss_r','epoch_loss_s','epoch_loss_pre','lr']
    #         writer.add_scalar(tags[0],epoch_loss,epoch)
    #         # writer.add_scalar(tags[1],epoch_loss_r,epoch)
    #         writer.add_scalar(tags[2],epoch_loss_s,epoch)
    #         writer.add_scalar(tags[3],epoch_loss_pre,epoch)
    #         writer.add_scalar(tags[4],optimizer.param_groups[0]["lr"],epoch)
    #         # print('Epoch:', epoch, 'Loss:', epoch_loss, 'Loss_r:', epoch_loss_r, 'Loss_s:', epoch_loss_s, 'Loss_pre:',
    #         #       epoch_loss_pre)
    #         print('Epoch:', epoch, 'Loss:', epoch_loss,'Loss_s:', epoch_loss_s, 'Loss_pre:',
    #               epoch_loss_pre)
    #
    #         print("------------------------------------------verification start----------------------------------------------------------")
    #         # 1111111111111111111111
    #
    #         if epoch % 10 == 0:
    #             early_stopping(epoch_loss, model)
    #             if early_stopping.early_stop:
    #                 print("Early stopping")
    #                 break
    #
    #         # 加验证集
    #         '''
    #         model.eval()
    #         if epoch % 10 == 0:
    #             dev_loss = 0
    #
    #             dev_p5 = 0
    #             dev_p10 = 0
    #             dev_p15 = 0
    #             dev_p20 = 0
    #
    #             dev_r5 = 0
    #             dev_r10 = 0
    #             dev_r15 = 0
    #             dev_r20 = 0
    #
    #             dev_f1_5 = 0
    #             dev_f1_10 = 0
    #             dev_f1_15 = 0
    #             dev_f1_20 = 0
    #             batch_num = len(dev_loader)
    #             batch_no = int(np.ceil(num_Pdev / batch_num))
    #             for j, dev_batch in enumerate(dev_loader):
    #                 _, _, _ = dev_batch
    #                 start = j * batch_no
    #                 end = min((j + 1) * batch_no, num_Pdev)
    #                 ps = ps_dev_onehot[start:end]
    #                 ph = ph_dev_onehot[start:end]
    #                 ps = torch.tensor(ps).to(device)
    #                 ph = torch.tensor(ph).to(device)
    #                 prediction_result = model(None, None, None, None, ps, test=True)
    #
    #                 dev_loss += criterion(prediction_result, ph).item()
    #                 # print(f"test_loss:{test_loss}")
    #
    #                 # 对预测结果得到降序排列后的元素索引
    #                 sorted_prediction = np.argsort(-prediction_result.cpu().detach().numpy(), axis=1)
    #                 for i, hid in enumerate(ph.cpu()):
    #                     trueLabel = np.where(hid == 1)[0].tolist()
    #                     result = sorted_prediction[i]
    #
    #                     # 前5个
    #                     top5 = result[:5].tolist()
    #                     # 使用集合的交集操作来获取交集数据
    #                     matched_herb = set(trueLabel) & set(top5)
    #                     count = len(matched_herb)
    #                     # 获取交集数据的个数
    #                     dev_p5 += count / 5
    #                     dev_r5 += count / len(trueLabel)
    #
    #                     # 前10个
    #                     top10 = result[:10]
    #                     # 使用集合的交集操作来获取交集数据
    #                     matched_herb = set(trueLabel) & set(top10)
    #                     count = len(matched_herb)
    #                     # 获取交集数据的个数
    #                     dev_p10 += count / 10
    #                     dev_r10 += count / len(trueLabel)
    #
    #                     # 前15个
    #                     top15 = result[:15]
    #                     # 使用集合的交集操作来获取交集数据
    #                     matched_herb = set(trueLabel) & set(top15)
    #                     count = len(matched_herb)
    #                     # 获取交集数据的个数
    #                     dev_p15 += count / 15
    #                     dev_r15 += count / len(trueLabel)
    #
    #                     # 前20个
    #                     top20 = result[:20]
    #                     # 使用集合的交集操作来获取交集数据
    #                     matched_herb = set(trueLabel) & set(top20)
    #                     count = len(matched_herb)
    #                     # 获取交集数据的个数
    #                     dev_p20 += count / 20
    #                     dev_r20 += count / len(trueLabel)
    #
    #             print(
    #                 "------------------------------------------verification finish----------------------------------------------------------")
    #
    #             print('dev_loss: ', dev_loss / len(dev_loader))
    #
    #             print('p5-10-20:', dev_p5 / num_Pdev, dev_p10 / num_Pdev, dev_p15 / num_Pdev, dev_p20 / num_Pdev)
    #             print('r5-10-20:', dev_r5 / num_Pdev, dev_r10 / num_Pdev, dev_r15 / num_Pdev, dev_r20 / num_Pdev)
    #
    #             print('f1_5-10-20: ',
    #                   2 * (dev_p5 / num_Pdev) * (dev_r5 / num_Pdev) / ((dev_p5 / num_Pdev) + (dev_r5 / num_Pdev)),
    #                   2 * (dev_p10 / num_Pdev) * (dev_r10 / num_Pdev) / ((dev_p10 / num_Pdev) + (dev_r10 / num_Pdev)),
    #                   2 * (dev_p15 / num_Pdev) * (dev_r15 / num_Pdev) / ((dev_p15 / num_Pdev) + (dev_r15 / num_Pdev)),
    #                   2 * (dev_p20 / num_Pdev) * (dev_r20 / num_Pdev) / ((dev_p20 / num_Pdev) + (dev_r20 / num_Pdev)))
    #
    #             early_stopping(dev_loss / len(dev_loader), model)
    #             if early_stopping.early_stop:
    #                 print("Early stopping")
    #                 break
    #             '''
    #
    #     writer.close()
    #     print(
    #         "------------------------------------------------test start----------------------------------------------------------")
    #     # model.load_state_dict(torch.load('checkpoint.pt'))
    #
    #     test_loss = 0
    #
    #     test_p5 = 0
    #     test_p10 = 0
    #     test_p15 = 0
    #     test_p20 = 0
    #
    #     test_r5 = 0
    #     test_r10 = 0
    #     test_r15 = 0
    #     test_r20 = 0
    #
    #     test_f1_5 = 0
    #     test_f1_10 = 0
    #     test_f1_15 = 0
    #     test_f1_20 = 0
    #     model.eval()
    #
    #     batch_num = len(test_loader)
    #     batch_no = int(np.ceil(num_Ptest / batch_num))
    #     output_syms = []
    #     output_groundtruth = []
    #     output_predictherbs = []
    #     output_precision = []
    #     for j, test_batch in enumerate(test_loader):
    #         _, _, _ = test_batch
    #         start = j * batch_no
    #         end = min((j + 1) * batch_no, num_Ptest)
    #         ps = ps_test_onehot[start:end]
    #         ph = ph_test_onehot[start:end]
    #         ps = torch.tensor(ps).to(device)
    #         ph = torch.tensor(ph).to(device)
    #         prediction_result = model(None, None, None, None, ps, test=True)
    #
    #         test_loss += criterion(prediction_result, ph).item()
    #         # print(f"test_loss:{test_loss}")
    #
    #         # 对预测结果得到降序排列后的元素索引
    #         sorted_prediction = np.argsort(-prediction_result.cpu().detach().numpy(), axis=1)
    #         for i, (hid, sid) in enumerate(zip(ph.cpu(), ps.cpu())):
    #             syms = np.where(sid == 1)[0].tolist()
    #             output_syms.append(syms)
    #
    #             trueLabel = np.where(hid == 1)[0].tolist()
    #             output_groundtruth.append(trueLabel)
    #             result = sorted_prediction[i]
    #
    #             # ************定性结果分析
    #             predict_herbs = result[:len(trueLabel)].tolist()
    #             output_predictherbs.append(predict_herbs)  # 真实处方取多少个，我的预测处方就取多少个
    #             matched_herb = set(trueLabel) & set(predict_herbs)
    #             count = len(matched_herb)
    #             output_precision.append([count / len(trueLabel)])
    #             # ************定性结果分析
    #
    #             # 前5个
    #             top5 = result[:5].tolist()
    #             # 使用集合的交集操作来获取交集数据
    #             matched_herb = set(trueLabel) & set(top5)
    #             count = len(matched_herb)
    #             # 获取交集数据的个数
    #
    #             test_p5 += count / 5
    #             test_r5 += count / len(trueLabel)
    #
    #             # 前10个
    #             top10 = result[:10]
    #             # 使用集合的交集操作来获取交集数据
    #             matched_herb = set(trueLabel) & set(top10)
    #             count = len(matched_herb)
    #             # 获取交集数据的个数
    #             test_p10 += count / 10
    #             test_r10 += count / len(trueLabel)
    #
    #             # 前15个
    #             top15 = result[:15]
    #             # 使用集合的交集操作来获取交集数据
    #             matched_herb = set(trueLabel) & set(top15)
    #             count = len(matched_herb)
    #             # 获取交集数据的个数
    #             test_p15 += count / 15
    #             test_r15 += count / len(trueLabel)
    #
    #             # 前20个
    #             top20 = result[:20]
    #             # 使用集合的交集操作来获取交集数据
    #             matched_herb = set(trueLabel) & set(top20)
    #             count = len(matched_herb)
    #             # 获取交集数据的个数
    #             test_p20 += count / 20
    #             test_r20 += count / len(trueLabel)
    #
    #     print('test_loss: ', test_loss / len(test_loader))
    #
    #     print('p5-10-20:', test_p5 / num_Ptest, test_p10 / num_Ptest, test_p15 / num_Ptest, test_p20 / num_Ptest)
    #     print('r5-10-20:', test_r5 / num_Ptest, test_r10 / num_Ptest, test_r15 / num_Ptest, test_r20 / num_Ptest)
    #
    #     print('f1_5-10-20: ',
    #           2 * (test_p5 / num_Ptest) * (test_r5 / num_Ptest) / ((test_p5 / num_Ptest) + (test_r5 / num_Ptest)),
    #           2 * (test_p10 / num_Ptest) * (test_r10 / num_Ptest) / ((test_p10 / num_Ptest) + (test_r10 / num_Ptest)),
    #           2 * (test_p15 / num_Ptest) * (test_r15 / num_Ptest) / ((test_p15 / num_Ptest) + (test_r15 / num_Ptest)),
    #           2 * (test_p20 / num_Ptest) * (test_r20 / num_Ptest) / ((test_p20 / num_Ptest) + (test_r20 / num_Ptest)))
    #
    #     precision.append([test_p5 / num_Ptest, test_p10 / num_Ptest, test_p15 / num_Ptest, test_p20 / num_Ptest])
    #     recall.append([test_r5 / num_Ptest, test_r10 / num_Ptest, test_r15 / num_Ptest, test_r20 / num_Ptest])
    #     f1score.append(
    #         [2 * (test_p5 / num_Ptest) * (test_r5 / num_Ptest) / ((test_p5 / num_Ptest) + (test_r5 / num_Ptest)),
    #          2 * (test_p10 / num_Ptest) * (test_r10 / num_Ptest) / ((test_p10 / num_Ptest) + (test_r10 / num_Ptest)),
    #          2 * (test_p15 / num_Ptest) * (test_r15 / num_Ptest) / ((test_p15 / num_Ptest) + (test_r15 / num_Ptest)),
    #          2 * (test_p20 / num_Ptest) * (test_r20 / num_Ptest) / ((test_p20 / num_Ptest) + (test_r20 / num_Ptest))
    #          ])
    #     print(
    #         "------------------------------------------test finish----------------------------------------------------------")
    # print(f"precision_{args.loop}:{precision}")
    # print(f"recall_{args.loop}:{recall}")
    # print(f"f1score_{args.loop}:{f1score}")
    # precision_mean = np.mean(np.array(precision), axis=0)
    # print(f"precision_mean:{precision_mean}")
    # recall_mean = np.mean(np.array(recall), axis=0)
    # print(f"recall_mean:{recall_mean}")
    # f1score_mean = np.mean(np.array(f1score), axis=0)
    # print(f"f1score_mean:{f1score_mean}")
    #
    # output_syms = np.array(output_syms, dtype=object).reshape(-1, 1)
    # output_groundtruth = np.array(output_groundtruth, dtype=object).reshape(-1, 1)
    # output_predictherbs = np.array(output_predictherbs, dtype=object).reshape(-1, 1)
    # output_precision = np.array(output_precision, dtype=object).reshape(-1, 1)
    # output_result = np.concatenate((output_syms, output_groundtruth, output_predictherbs, output_precision), axis=1)
    # data_result = pd.DataFrame(output_result, columns=['syms', 'groundtruth', 'predictherbs', 'output_precision'])
    # data_result.to_excel('output_result.xlsx')
