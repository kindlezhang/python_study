import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="./GNN/data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {0} dataset...'.format(dataset)) # 0表示format的第一个参数


    # 特征提取出来的原始数据，[节点ID, 特征1, 特征2, ..., 特征M, 标签] 
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str)) # 导入node数据，表示把每一列元素都按字符串（str）类型导入
    # print(idx_features_labels)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # numpy的切片功能，取特征feature，再压缩成稀疏矩阵
    labels = encode_onehot(idx_features_labels[:, -1])  # one-hot label， 把每个类别标签转换成 0-1 向量

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # 节点
    idx_map = {j: i for i, j in enumerate(idx)}   # 构建节点的索引字典
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),  # 导入edge的数据
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)    # 将之前的转换成字典编号后的边
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),  # 构建边的邻接矩阵
                        shape=(labels.shape[0], labels.shape[0]), # 矩阵行数。对于特征矩阵，shape[0] 就是样本/节点/图的数目。 对于邻接矩阵，shape[0] 就是节点数。
                        dtype=np.float32)

    # build symmetric adjacency matrix，计算转置矩阵。将有向图转成无向图
    # adj.T 是转置矩阵， adj.T > adj 是布尔矩阵, adj.T.multiply(adj.T > adj) 是取转置矩阵中大于原矩阵的元素, 再加上原矩阵，得到对称矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 归一化邻接矩阵涉及度归一化、加自环，是为了让每个节点在信息聚合时权重平均分配，缓解度数不同导致的信息传播不均，以及提升稳定性和表达能力。
    features = normalize(features)   # 对特征做了归一化的操作。并不是必要的
    adj = normalize(adj + sp.eye(adj.shape[0]))   # 对A+I归一化，GCN是必要的
    # 训练，验证，测试的样本
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    # 将numpy的数据转换成torch格式
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  #  矩阵行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.   # 如果是inf，转换成0
    r_mat_inv = sp.diags(r_inv)  # 构造对角矩阵
    mx = r_mat_inv.dot(mx)  # 构造D-1*A，非对称方式，简化方式
    # 这是**“非对称”归一化**，因为只施加左边的 D⁻¹，没有对称归一化中常见的 D^(-1/2) A D^(-1/2)

    # r_inv_sqrt = np.power(rowsum, -0.5).flatten() 
    # r_inv[np.isinf(r_inv)] = 0.   
    # r_mat_inv = sp.diags(r_inv_sqrt)  
    # mx.dot(r_mat_inv).transpose().dot(r_mat_inv)  # D^(-1/2) * A * D^(-1/2)，对称归一化
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
