import torch
import torch.nn as nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, InnerProductInteraction
from typing import List
from functools import reduce


class DeeperCosineSim(nn.Module):
    """点积注意力机制"""
    def __init__(self, embedding_dim, net_dropout):
        super(DeeperCosineSim, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = net_dropout
        
        self.deeper_x1 = nn.Sequential(
            nn.Linear(self.embedding_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, self.embedding_dim)
        )
        self.deeper_x2 = nn.Sequential(
            nn.Linear(self.embedding_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, self.embedding_dim)
        )
        self.cosine_sim = nn.CosineSimilarity(dim=1)
        
    def forward(self, x1, x2):
        x1 = self.deeper_x1(x1)
        x2 = self.deeper_x2(x2)
        
        return self.cosine_sim(x1, x2)



class UIMEF_CosineSim_deeper(BaseModel):
    """ The FwFM model
        Reference:
          - Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising, WWW'2018.
    """
    def __init__(self, 
                 feature_map, 
                 model_id="UIMEF_dotMLP", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=16, 
                 net_dropout=0.1,
                 regularizer=None, 
                 **kwargs):
        """ 
        linear_type: `LW`, `FeLV`, or `FiLV`
        """
        super(UIMEF_CosineSim_deeper, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   net_dropout=net_dropout,
                                   embedding_regularizer=regularizer, 
                                   net_regularizer=regularizer,
                                   **kwargs) 
        self.local_feature_map = feature_map.features
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        self.matching_layer = nn.ModuleList()
        for attr in self.local_feature_map.keys():
            if attr == 'UserID' or attr == 'timestamp':
                continue
            else:
                self.matching_layer.append(DeeperCosineSim(embedding_dim, net_dropout))
        item_all_dim = embedding_dim * (len(self.local_feature_map) - 2)
        self.matching_layer.append(nn.Sequential(
                    nn.Linear(item_all_dim, 64),
                    nn.LayerNorm(64),
                    nn.ReLU(),
                    nn.Dropout(net_dropout),
                    nn.Linear(64, 64),
                    nn.LayerNorm(64),
                    nn.ReLU(),
                    nn.Dropout(net_dropout),
                    nn.Linear(64, 64),
                    nn.LayerNorm(64),
                    nn.ReLU(),
                    nn.Dropout(net_dropout),
                    nn.Linear(64, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    nn.ReLU(),
        ))

        output_all_dim = 3 * embedding_dim + len(self.matching_layer) - 1
        self.output_layer = nn.Sequential(
            nn.Linear(output_all_dim, 64),
            nn.ReLU(),
            nn.Dropout(net_dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(net_dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(net_dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(net_dropout),
            nn.Linear(8, 1)
        )

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        inputs dim: [batch_size, num_features, embed_dim]
        在feature_map中，要求UserID排在第一位，timestamp在最后一位
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)

        match_output = list()
        for i in range(1, feature_emb.shape[1]-1):
            match_output.append(
                self.matching_layer[i-1](feature_emb[:,0,:], feature_emb[:,i,:]).reshape(-1,1)
            )
        match_output.append(
            self.matching_layer[-1](torch.cat([feature_emb[:,i,:] for i in range(1, feature_emb.shape[1]-1)], dim=1))
        )
        y_pred = self.output_layer(torch.cat(match_output + [feature_emb[:,0,:], feature_emb[:,-1,:]], dim=1))

        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": (y_pred, feature_emb[:,0,:], feature_emb[:,1:-1,:])}
        return return_dict
