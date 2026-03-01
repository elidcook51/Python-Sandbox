import numpy as np





# def compute_fibonacci(n):
#     if n == 1:
#         return [0]
#     if n == 2:
#         return [0,1]
#     startList = [0,1]
#     curSpot = 1
#     for i in range(n):
#         startList.append(startList[curSpot] + startList[curSpot - 1])
#         curSpot += 1
#     return startList
# num = int(input("Enter a integer between 1 and 100: "))
# if num > 100 or num < 1:
#     print("Please enter a number between 1 and 100")
# else:
#     print(compute_fibonacci(num))


# import pandas as pd
# import networkx as nx
# import numpy as np
# import matplotlib.pyplot as plt

# import plotly.graph_objects as go
# import plotly.express as px
# import plotly.figure_factory as ff
# import matplotlib as mpl

# G = nx.karate_club_graph()

# deg = dict(G.degree())
# sizes = [deg[n] * 75 for n in G.nodes()]

# nx.draw(G, node_size = sizes, with_labels=True)
# plt.savefig("C:/Users/ucg8nb/Downloads/Graph of Graph.png")

# curMax = 0
# curNode = 0
# for n in G.nodes():
#     if G.degree(n) > curMax:
#         curMax = G.degree(n)
#         curNode = n

# G.remove_node(curNode)
# print(nx.is_connected(G))

# neighbors = list(G.neighbors(33))
# print(neighbors)

# degrees = [d for _, d in G.degree()]
# plt.hist(degrees, bins = 10)
# plt.title("Histogram of Degrees")
# plt.xlabel("Degree")
# plt.ylabel("Count of degree")
# plt.savefig("C:/Users/ucg8nb/Downloads/Average Degree Figure.png")

# avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()

# nx.draw(G, with_labels = True)
# plt.show()