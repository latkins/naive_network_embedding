from graph import Graph
from model import Net
import torch.optim as optim
from tqdm import tqdm


def main():
    #graph_pos = '/users2/xfsun/node2vec/graph/karate.edgelist'
    #graph_pos = '/users2/xfsun/m10_data/edge_list_0.data'
    graph_pos = '/users2/xfsun/zhihu_data/cikm/data/edge_list_0.data'
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
            #loss.backward()
            #optimizer.step()
            running_loss += loss.data[0]
        #print(running_loss)
        running_loss=0.0
        # net.save_embedding(['1'], graph)


if __name__ == '__main__':
    main()
