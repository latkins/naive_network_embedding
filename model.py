import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CosineEmbeddingLoss


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
        self.loss_function=CosineEmbeddingLoss(margin=1.0)

    def forward(self, edge, negative_edges):
        emb_u = self.u_embeddings(Variable(torch.LongTensor([edge.u])))
        emb_v = self.v_embeddings(Variable(torch.LongTensor([edge.v])))
        #score = torch.dot(emb_u, emb_v)
        #score = F.logsigmoid(score)
        score=self.loss_function(emb_u, emb_v, Variable(torch.LongTensor([1])))
        scores = [score]
        assert len(negative_edges)<=20
        for edge in negative_edges:
            emb_u = self.u_embeddings(Variable(torch.LongTensor([edge.u])))
            emb_v = self.v_embeddings(Variable(torch.LongTensor([edge.v])))
            score=self.loss_function(emb_u, emb_v, Variable(torch.LongTensor([-1])))
            #score = torch.dot(emb_u, emb_v)
            #score = F.logsigmoid(-1 * score)
            scores.append(score)
        # print(scores)
        loss = -1 * sum(scores)
        return loss

    # def save_embedding(self, to_be_saved_node_name, graph):
    #     for node_name in to_be_saved_node_name:
    #         nid = graph.name_to_id[node_name]
    #         nid = Variable(torch.LongTensor([nid]))
    #         emb = self.embeddings(nid).data[0]
    #         print(emb)
