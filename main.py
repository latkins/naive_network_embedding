from graph import Graph
from model import Net
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys

def main():
    graph_pos = sys.argv[1]
    graph = Graph(graph_pos)
    emb_size = len(graph.graph)
    emb_dimension = 100
    net = Net(emb_size, emb_dimension)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    batch_size=10000
    for i in range(100):
        running_loss = 0.0
        for batch_index in tqdm(range(len(graph.edge_list)/batch_size)):
            begin=batch_index*batch_size
            end=(batch_index+1)*batch_size
            positive_edges=graph.edge_list[begin:end]
            negative_edges = graph.negative_sampling(positive_edges, 20)
            pos_u=[edge.u for edge in positive_edges]
            pos_v=[edge.v for edge in positive_edges]
            neg_u=[edge.u for edge in negative_edges]
            neg_v=[edge.v for edge in negative_edges]
            optimizer.zero_grad()
            loss = net.forward(pos_u,pos_v, neg_u, neg_v)
            loss.backward()
            optimizer.step()
            #running_loss += loss.data[0]
        #print(running_loss)
        #running_loss=0.0


if __name__ == '__main__':
    main()
