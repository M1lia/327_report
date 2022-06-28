class Graph:

    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot


        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def Kruskal_MST(self):

        result = []

        i = 0

        e = 0

        self.graph = sorted(self.graph,
                            key=lambda item: item[2])

        parent = []
        rank = []

        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        while e < self.V - 1:

            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)


        out_lst_ = []
        for u, v, weight in result:

            out_lst_.append([u, v, weight])
        return (out_lst_)


input_lst = []
N, M = map(int, input().split())
g = Graph(N)
for i in range(M):

    a, b, c = (int(i) for i in input(">").split())
    input_lst.append([a - 1, b - 1, c])
    g.add_edge(a - 1, b - 1, c)

out_lst = []
out_lst = g.Kruskal_MST()
last_out = []

for i in out_lst:
    if i in input_lst:
        last_out.append(input_lst.index(i))
print(*last_out)
