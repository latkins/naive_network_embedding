import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        '''
        emb_size: the count of nodes which have embedding
        emb_dimension: embedding dimention
        '''
        super(Net, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    @profile
    def forward(self, edge, negative_edges):
        losses=[]
        emb_u = self.u_embeddings(Variable(torch.LongTensor([edge.u])))
        emb_v = self.v_embeddings(Variable(torch.LongTensor([edge.v])))
        score = torch.dot(emb_u, emb_v)
        score = F.logsigmoid(score)
        losses.append(-1*score)
        for edge in negative_edges:
            neg_emb_u = self.u_embeddings(Variable(torch.LongTensor([edge.u])))
            neg_emb_v = self.v_embeddings(Variable(torch.LongTensor([edge.v])))
            neg_score = torch.dot(neg_emb_u, neg_emb_v)
            neg_score = F.logsigmoid(neg_score)
            losses.append(-1*neg_score)
        return sum(losses)

