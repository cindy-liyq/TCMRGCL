import numpy as np
from sklearn.decomposition import NMF

# 创建一个非负矩阵
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# 初始化NMF模型，设置分解后的低秩为2
nmf = NMF(n_components=2, init='random', random_state=42)

# 对矩阵X进行分解
W = nmf.fit_transform(X)  # 得到基矩阵
H = nmf.components_       # 得到系数矩阵

print("原始矩阵 X：\n", X)
print("\n基矩阵 W：\n", W)
print("\n系数矩阵 H：\n", H)
print("\n重构的矩阵（近似）：\n", np.dot(W, H))
