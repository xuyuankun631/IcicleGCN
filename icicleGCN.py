from __future__ import print_function, division
import argparse
import os
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear

from save_icicleGCN_model import save_model
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation_GraphNetwork import eva
from collections import Counter


# torch.cuda.set_device(1)


class AE(nn.Module):  # autoencoder

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))  # decoder
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3).softmax(dim=1)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class IcicleGCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):
        super(IcicleGCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        net = self.ae
        model_new_dict = net.state_dict()
        checkpoint = torch.load(args.model_path)
        state_dict = {}
        for k, vv in checkpoint['net'].items():
            state_dict[k[4:]] = vv
        state_dict = {k: vv for k, vv in state_dict.items() if k in model_new_dict.keys()}
        model_new_dict.update(state_dict)
        self.ae.load_state_dict(model_new_dict)

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj1, adj2):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        sigma = 0.5
        gama = 0.2

        # GCN Module
        h11 = self.gnn_1(x, adj1)
        h21 = self.gnn_1(x, adj2)

        h12 = self.gnn_2(sigma * h11 + gama * h21 + sigma * tra1, adj1)
        h22 = self.gnn_2(sigma * h21 + gama * h11 + sigma * tra1, adj2)

        h13 = self.gnn_3(sigma * h12 + gama * h22 + sigma * tra2, adj1)
        h23 = self.gnn_3(sigma * h22 + gama * h12 + sigma * tra2, adj2)

        h14 = self.gnn_4(sigma * h13 + gama * h23 + sigma * tra3, adj1)
        h24 = self.gnn_4(sigma * h23 + gama * h13 + sigma * tra3, adj2)

        h15 = self.gnn_5(sigma * h14 + gama * h24 + sigma * z, adj1, active=False)
        h25 = self.gnn_5(sigma * h24 + gama * h14 + sigma * z, adj2, active=False)

        predict1 = F.softmax(h15, dim=1)
        predict2 = F.softmax(h25, dim=1)

        # Joint Self-Adaptive Learning Mechanism
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict1, predict2, z


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_icicleGCN(dataset):
    model = IcicleGCN(500, 500, 2000, 2000, 500, 500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph 1
    adj1 = load_graph(args.name, 1)
    adj1 = adj1.cuda()

    # KNN Graph 2
    adj2 = load_graph(args.name, 15)
    adj2 = adj2.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')  # evaluation

    fname = 'icicleGCN_result/ImageNet-10/{}_1_.txt'.format(args.name)

    f = open(fname, 'w')
    epoch_list = []
    Q_list = []
    Z1_list = []
    Z2_list = []
    P_list = []

    for epoch in range(1000):
        if epoch % 1 == 0:
            # update_interval
            _, tmp_q, pred1, pred2, _ = model(data, adj1, adj2)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)  #

            epoch_list.append(epoch)

            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            res2 = pred1.data.cpu().numpy().argmax(1)  # Z
            res21 = pred2.data.cpu().numpy().argmax(1)  # Z2
            res3 = p.data.cpu().numpy().argmax(1)  # P

            a1, g1 = eva(y, res1, str(epoch) + 'Q')
            a2, g2 = eva(y, res2, str(epoch) + 'Z')
            a3, g3 = eva(y, res21, str(epoch) + 'Z2')
            a4, g4 = eva(y, res3, str(epoch) + 'P')

            Q_list.append(g1.item())
            Z1_list.append(g2.item())
            Z2_list.append(g3.item())
            P_list.append(g4.item())

            f.write(a1)
            f.write('\n')
            f.write(a2)
            f.write('\n')
            f.write(a3)
            f.write('\n')
            f.write(a4)
            f.write('\n')

        x_bar, q, pred1, pred2, _ = model(data, adj1, adj2)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss1 = F.kl_div(pred1.log(), p, reduction='batchmean')
        ce_loss2 = F.kl_div(pred2.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.1 * kl_loss + 0.05 * ce_loss1 + 0.01 * ce_loss2 + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            save_model(args, model, epoch + 1)
    f.close()
    save_model(args, model, epoch + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='imageNet-10')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)  # 1e-3
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--model_path', type=str, default='tar')
    parser.add_argument('--test_path', type=str, default='tar')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    dataset = load_data(args.name)

    if args.name == 'cifar100':
        args.n_clusters = 20
        args.n_z = 20
        args.n_input = 512
        # args.model_path = ''

    if args.name == 'cifar10':
        args.n_clusters = 10
        args.n_input = 512
        # args.model_path = ''

    if args.name == 'imageNet-dogs':
        args.n_clusters = 15
        args.n_z = 15
        args.n_input = 512
        # args.model_path = ''
        # args.test_path = ''

    if args.name == 'imageNet-10':
        args.n_clusters = 10
        args.n_z = 10
        args.n_input = 512
        args.model_path = 'save/ImageNet-10/checkpoint_1015.tar'
        args.test_path = 'save_icicleGCN/ImageNet-10'



    print(args)
    train_icicleGCN(dataset)
