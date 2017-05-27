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
        self.embeddings = nn.Embedding(emb_size, emb_dimension)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.emb_dimension
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, edge, negative_edges):
        emb_u = self.embeddings(Variable(torch.LongTensor([edge.u])))
        emb_v = self.embeddings(Variable(torch.LongTensor([edge.v])))
        score = torch.dot(emb_u, emb_v)
        score = F.logsigmoid(score)
        scores = [score]
        for edge in negative_edges:
            emb_u = self.embeddings(Variable(torch.LongTensor([edge.u])))
            emb_v = self.embeddings(Variable(torch.LongTensor([edge.v])))
            score = torch.dot(emb_u, emb_v)
            score = F.logsigmoid(-1 * score)
            scores.append(score)
        # print(scores)
        loss = -1 * sum(scores)
        return loss

    def save_embedding(self, to_be_saved_node_name, graph):
        for node_name in to_be_saved_node_name:
            nid = graph.name_to_id[node_name]
            nid = Variable(torch.LongTensor([nid]))
            emb = self.embeddings(nid).data[0]
            print(emb)
