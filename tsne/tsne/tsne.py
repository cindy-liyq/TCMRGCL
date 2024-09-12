import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import seaborn as sns


    
if __name__ == '__main__':
    # load data from pickle
    #参数设置
    dpi = 300
    hreb_color = '#ED7370'
    sym_color = '#99ACCB'


    best_hreb_emb = pickle.load(open('best_noCPreT_herb_emb.pkl', 'rb')).cpu().detach().numpy()
    best_sym_emb = pickle.load(open('best_noCPreT_sym_emb.pkl', 'rb')).cpu().detach().numpy()


    # TSNE for hreb
    tsne_hreb = TSNE(n_components=2, random_state=0,verbose=1,perplexity=30)
    best_hreb_emb = tsne_hreb.fit_transform(best_hreb_emb)
    # TSNE for sym
    tsne_sym = TSNE(n_components=2, random_state=0, verbose=1,perplexity=30)
    best_sym_emb = tsne_sym.fit_transform(best_sym_emb)

    #scatter 
    plt.figure(figsize=(10, 10),dpi=dpi)
    plt.scatter(best_hreb_emb[:, 0], best_hreb_emb[:, 1], s=10)
    plt.title('TSNE for HERB')
    plt.savefig('scatter_tsne_noCPreT_herb.png')
    plt.show()

    plt.figure(figsize=(10, 10),dpi=dpi)
    plt.scatter(best_sym_emb[:, 0], best_sym_emb[:, 1], s=10)
    plt.title('TSNE for Sym')
    plt.savefig('scatter_tsne_noCPreT_sym.png')
    plt.show()

    ##kdeplot
    plt.figure(figsize=(10, 10),dpi=dpi)
    sns.kdeplot(x=best_hreb_emb[:, 0], y=best_hreb_emb[:, 1], shade=True,fill=True,kernel='gau',bw_adjust=0.7,color=hreb_color)
    plt.title('TSNE for HERB')
    plt.savefig('kde_tsne_noCPreT_herb.png')
    plt.show()
    
    plt.figure(figsize=(10, 10),dpi=dpi)
    sns.kdeplot(x=best_sym_emb[:, 0],y=best_sym_emb[:, 1], shade=True,fill=True,kernel='gau',bw_adjust=0.7,color=sym_color)
    plt.title('TSNE for Sym')
    plt.savefig('kde_tsne_noCPreT_sym.png')
    plt.show()


