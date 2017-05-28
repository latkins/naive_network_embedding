import random
random.seed(12345)


class Edge:
    def __init__(self, u, v):
        self.u = u
        self.v = v


class Graph:
    def __init__(self, fname):
        self.name_to_id = dict()
        self.id_to_name = dict()
        self.graph = []
        self.edge_list = []
        f = open(fname)
        f.readline()
        for line in f:
            line = line.strip().split(' ')
            if line[0] not in self.name_to_id:
                nid = len(self.name_to_id)
                self.name_to_id[line[0]] = nid
                self.id_to_name[nid] = line[0]
                self.graph.append(set())
            if line[1] not in self.name_to_id:
                nid = len(self.name_to_id)
                self.name_to_id[line[1]] = nid
                self.graph.append(set())
                self.id_to_name[nid] = line[1]
            nid0 = self.name_to_id[line[0]]
            nid1 = self.name_to_id[line[1]]
            self.graph[nid0].add(nid1)
            self.edge_list.append(Edge(nid0, nid1))

    def negative_sampling(self, edge, count):
        neg_edges = []
        for _ in range(count):
            neg_v = random.randint(0, len(self.graph) - 1)
            #if neg_v not in self.graph[edge.u]:
            neg_edges.append(Edge(edge.u, neg_v))
        return neg_edges


def test():
    a = Graph('/Users/sunxiaofei/workspace/node2vec/graph/karate.edgelist')


if __name__ == '__main__':
    test()
