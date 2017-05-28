from graph import Graph
from model import Net
import torch.optim as optim
from tqdm import tqdm


def main():
    graph_pos = '/Users/sunxiaofei/workspace/node2vec/graph/karate.edgelist'
    graph = Graph(graph_pos)
    emb_size = len(graph.graph)
    emb_dimension = 100
    net = Net(emb_size, emb_dimension)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for i in tqdm(range(10000)):
        running_loss = 0.0
        for edge in graph.edge_list:
            # print(edge.u, edge.v)
            neg_edges = graph.negative_sampling(edge, 20)
            optimizer.zero_grad()
            loss = net.forward(edge, neg_edges)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
        # print(running_loss)
        # net.save_embedding(['1'], graph)


if __name__ == '__main__':
    main()
