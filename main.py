from graph import Graph
from model import Net
import torch.optim as optim
from tqdm import tqdm
import sys


def main():
    graph_pos = sys.argv[1]
    graph = Graph(graph_pos)
    emb_size = len(graph.graph)
    emb_dimension = 100
    net = Net(emb_size, emb_dimension)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for i in tqdm(range(10000)):
        running_loss = 0.0
        for edge in tqdm(graph.edge_list):
            neg_edges = graph.negative_sampling(edge, 20)
            optimizer.zero_grad()
            loss = net.forward(edge, neg_edges)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
        running_loss=0.0


if __name__ == '__main__':
    main()
