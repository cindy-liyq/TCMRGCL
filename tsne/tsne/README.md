## TSNE使用方法
其实T-SNE就是类似于PCA的一个降维技术，将你产生的embbding从一个高维降至2维，就可以作为xy轴的两个点可视化出来，因此，T-SNE是一个需要训练的模块。

### 流程

高维embedding -> T-SNE模型训练 ->2维度 ->Matplotlib绘图

### 环境准备

```bash
pip install seaborn
pip install scikit-learn
pip install matplotlib
pip install numpy
pip install pickle
```
将`best_hreb_emb.pkl`和`best_sym_emb.pkl`放到目录下
### 运行代码
```
python tsne.py
```
### 代码说明
会产生两组图
`kde_xxx.png` 和`scatter_xxx.png`
分别代表`核密度估计图`和`散点图`

