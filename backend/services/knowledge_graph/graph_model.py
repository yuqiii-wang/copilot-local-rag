import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

class KeywordReconstructionHGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_keywords):
        super(KeywordReconstructionHGNN, self).__init__()
        # Hypergraph Convolution
        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels, hidden_channels)
        
        # Output head: Predicts probability of each keyword being present in the document
        self.keyword_predictor = nn.Linear(hidden_channels, num_keywords)

    def forward(self, x, hyperedge_index):
        # x: Node features
        x = self.conv1(x, hyperedge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, hyperedge_index)
        x = F.relu(x) # Embedding
        
        # Predict keywords
        logits = self.keyword_predictor(x)
        return logits
