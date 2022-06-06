import torch.nn as nn
import torch
from torch.nn.functional import normalize
from torch.nn import Linear, Softmax
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(Encoder, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z,n_dec_1)
        self.dec_2 = Linear(n_dec_1,n_dec_2)
        self.dec_3 = Linear(n_dec_2,n_dec_3)
        self.x_bar_layer = Linear(n_dec_3,n_input)

    def forward(self,x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3).softmax(dim=1)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return z , x_bar


class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )

        self.enc = Encoder(
            n_enc_1 = 500,
            n_enc_2 = 500,
            n_enc_3 = 2000,
            n_dec_1 = 2000,
            n_dec_2 = 500,
            n_dec_3 = 500,
            n_input = self.resnet.rep_dim,
            n_z = self.cluster_num
        )

    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i,bar_i = self.enc(h_i)
        c_j, _     = self.enc(h_i)

        return z_i, z_j, c_i, c_j, h_i, bar_i


    def forward_cluster(self, x):

        h = self.resnet(x)
        c,_ = self.enc(h)
        c = torch.argmax(c, dim=1)
        return c

        # # to capture the feature matrix of images
        # h, features_for_ensemble_cluster = self.resnet(x)
        # z = self.instance_projector(h)
        # z_n = normalize(self.instance_projector(h), dim=1)
        # z_copy = z.clone().detach()
        # z_n_copy = z.clone().detach()
        # features_for_ensemble_cluster.append(z_copy.view(z.size(0), -1))
        # features_for_ensemble_cluster.append(z_n_copy.view(z_n.size(0), -1))
        # c,_ = self.enc(h)
        # c = torch.argmax(c, dim=1)
        # return c, features_for_ensemble_cluster






