import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stop
from .transformer_layers import SelfAttnLayer
from .backbone import Backbone, Xie2019
from .utils import custom_replace,weights_init
from .graph_layers import GraphConvolution
from .position_enc import PositionEmbeddingSine,positionalencoding2d
from collections import OrderedDict
torch.set_printoptions(threshold=np.inf)
import gensim.models


class CTranModel(nn.Module):
    def __init__(self,num_labels,use_lmt,pos_emb=False,layers=3,heads=4,dropout=0.1,no_x_features=False):
        super(CTranModel, self).__init__()
        self.use_lmt = use_lmt
        
        self.no_x_features = no_x_features # (for no image features)

        self.backbone = Backbone()
        model = gensim.models.Word2Vec(corpus_file='./TrainEmbeddings.txt', vector_size=2048, window=2, min_count=1)

        hidden = 2048 # this should match the backbone output feature size

        self.downsample = False
        if self.downsample:
            self.conv_downsample = nn.Conv2d(hidden,hidden,(1,1))
        
        # Label Embeddings new
        word2vec = model.wv["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK"]
        self.label_lt = torch.tensor(word2vec)
        self.label_lt = torch.unsqueeze(self.label_lt, 0)

        # State Embeddings
        self.known_label_lt = nn.Embedding(3, hidden, padding_idx=0) 

        # Position Embeddings (for image features)
        self.use_pos_enc = pos_emb
        if self.use_pos_enc:
            self.position_encoding = positionalencoding2d(hidden, 18, 18).unsqueeze(0)

        # Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden,heads,dropout) for _ in range(layers)])

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        self.output_linear = nn.Linear(hidden,num_labels)

        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)


        self.known_label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)


        # GCN module
        from models.create_adjacency_matrix import normalize_adjacency_matrix
        self.adj_mat = np.load("./adjacency_matrices/adj_reweighted_mlgcn.npy")
        self.adj_mat = normalize_adjacency_matrix(self.adj_mat)
        self.adj_mat = torch.from_numpy(self.adj_mat)

        self.word_embeddings = np.load("./word_embeddings/one_hot.npy")
        self.word_embeddings = torch.from_numpy(self.word_embeddings)

        assert self.adj_mat.shape[0] == 17, "Number of classes does not match dimensionality of adjacency matrix: NUmber of classes: {}  Adjacency Matrix shape: {}".format(17, self.adj_mat.shape)
        assert self.word_embeddings.shape[0] == 17, "Number of classes does not match dimensionality of Word embedding matrix: NUmber of classes: {}  Word embedding Matrix shape: {}".format(17, self.word_embeddings.shape)
        assert self.adj_mat.shape[0] == self.adj_mat.shape[1], "Adjacency matrix is not square: {}".format(self.adj_mat.shape)

        self.adj_mat = self.adj_mat.float()
        self.word_embeddings = self.word_embeddings.float()

        self.lrelu_slope = 0.2
        self.gc1 = GraphConvolution(2048, 1024)
        self.gc2 = GraphConvolution(1024, 17)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.register_buffer("adj", self.adj_mat)
        self.register_buffer("word_embeddings2", self.word_embeddings)

        self.forward_gcn = GraphConvolution(2048, 2048)
        self.gcn_classifier = nn.Conv1d(2048, 17, 1)



    def forward(self,images,mask):
       
        init_label_embeddings = self.label_lt.repeat(images.size(0),1, 1)
        
        features = self.backbone(images)

        if self.downsample:
            features = self.conv_downsample(features)
        if self.use_pos_enc:
            pos_encoding = self.position_encoding(features,torch.zeros(features.size(0),18,18, dtype=torch.bool))
            features = features + pos_encoding

        features = features.view(features.size(0),features.size(1),-1).permute(0,2,1) 
        

        if self.use_lmt:
            # Convert mask values to positive integers for nn.Embedding
            label_feat_vec = custom_replace(mask,0,1,2).long()

            # Get state embeddings
            state_embeddings = self.known_label_lt(label_feat_vec)

            # Add state embeddings to label embeddings
            init_label_embeddings += state_embeddings

        # Input for GCN
        input_word_embeddings = init_label_embeddings.permute(0,2,1)

        if self.no_x_features:
            embeddings = init_label_embeddings 
        else:
            # Concat image and label embeddings
            embeddings = torch.cat((features,init_label_embeddings),1) 
 
        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)        
        attns = []
        for layer in self.self_attn_layers:
            embeddings,attn = layer(embeddings,mask=None)
            attns += attn.detach().unsqueeze(0).data
        
        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:,-init_label_embeddings.size(1):,:]  
        
        output = self.output_linear(label_embeddings) 
        
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0),1,1)
        output = (output*diag_mask).sum(-1) 
        

        # GCN module
        adj = self.adj_mat.repeat(images.size(0),1, 1)
        gcn = self.forward_gcn(adj, input_word_embeddings) + input_word_embeddings
        gcn_features = self.gcn_classifier(gcn) # torch.Size([64, 17, 17])
        gcn_features = (gcn_features*diag_mask).sum(-1)
        
        output = gcn_features# 0.8*output + 0.2*gcn_features
    

        return output

