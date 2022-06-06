import math
import torch
import torch.nn.functional as F
from torch.cuda import device
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GNNLayer(Module):    #特征数量和DNN编码器数量一直，保证学得的信息能线性运算
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features   #FloatTensor类型转换, 将list ,numpy转化为tensor的矩阵
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        #将权值矩阵随机初始化
        torch.nn.init.xavier_uniform_(self.weight)#一个服从均匀分布的Glorot初始化器

    def forward(self, features, adj, active=True):
        #                  H✖ W   特征矩阵和权值矩阵相乘
        support = torch.mm(features, self.weight) #矩阵a和b矩阵相乘
        #                   邻接矩阵A  ✖ H 乘以 W
        output = torch.spmm(adj, support)    #矩阵乘法
        # adj_=adj.to(device)
        # support_=support.to(device)
        # output = torch .spmm(adj_, support_)

        if active:#   将输出进行非线性激活
            output = F.relu(output)
        return output

