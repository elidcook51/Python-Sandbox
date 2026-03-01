import networkx
import numpy as np
import matplotlib.pyplot as plt

# A = np.array([
#     [0,1,1,1,1,0,0,1],
#     [1,0,1,0,0,0,0,1],
#     [1,1,0,1,0,0,0,0],
#     [1,0,1,0,1,0,0,0],
#     [1,0,0,1,0,1,1,0],
#     [0,0,0,0,1,0,1,0],
#     [0,0,0,0,1,1,0,1],
#     [1,1,0,0,0,0,1,0]], dtype = float)

# w, v = np.linalg.eig(A)

# idx = np.argmax(w)

# x = v[:, idx]
# x = np.abs(x)
# x = x / np.linalg.norm(x)

# length = np.sum(x)
# print(x / length)

lengths = [1, 2, 3, 4, 5, 6]
counts = [0, 1, 6, 6, 4, 2]

plt.bar(lengths, counts)
plt.xlabel("Lengths")
plt.ylabel("Number of that length")
plt.title("Distribution of Path Lengths between 2 and 5")
plt.savefig("C:/Users/ucg8nb/Downloads/Path Lengths.png")