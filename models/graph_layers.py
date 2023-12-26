import math
import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.relu = nn.LeakyReLU(0.2)
        self.weight = nn.Conv1d(in_features, out_features, 1)
        

    def forward(self, adj, node):
        nodes = torch.matmul(node, adj)
        nodes = self.relu(nodes)
        nodes = self.weight(nodes)
        nodes = self.relu(nodes)
        return nodes

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




class ConvGraphCombination(nn.Module):

    def __init__(self, merge_conv, gcn_act):
        super(ConvGraphCombination, self).__init__()
        self.merge_conv = merge_conv
        self.gcn_act = gcn_act

    def forward(self, cnn_feature, gcn_feature):

        cnn_input = cnn_feature #Copy of input, B x C x H x W

        cnn_feature = cnn_feature.transpose(1, 2) # B x H x C x W
        cnn_feature = cnn_feature.transpose(2, 3).contiguous() # B x H x W x C
        cnn_feature = cnn_feature.view(-1, cnn_feature.shape[-1]) # BHW x C

        gcn_feature = gcn_feature.transpose(0,1) # C x classes
        gcn_feature = self.gcn_act(gcn_feature) # C x classes

        cnn_feature = torch.matmul(cnn_feature, gcn_feature) # BHW x classes 
        cnn_feature = cnn_feature.view(cnn_input.shape[0], cnn_input.shape[2], cnn_input.shape[3], -1) # B x H x W x Classes
        cnn_feature = cnn_feature.transpose(2, 3) # B x H x Classes x W
        cnn_feature = cnn_feature.transpose(1, 2) # B x Classes x H x W

        cnn_feature = self.merge_conv(cnn_feature) # B x C x H x W
        cnn_feature += cnn_input # B x C x H x W

        return cnn_feature