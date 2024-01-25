# -*- coding: utf-8 -*-
"""
Created on Fri Sep 1  2023

@author: from OU Lab. FAH,SYSU 
Wangchen wch_bioinformatics@163.com
"""
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.decomposition import NMF

def nmf_cluster(mat,n_components=9,init = 'random', algo='bpp',use_gpu=True):#
    
    #from sklearn.datasets import fetch_20newsgroups
    from time import time
    n_samples = mat.shape[0]
    n_features =mat.shape[1]
    #n_components = 12
    #n_top_words = 20
    #batch_size = 28
    #init = "nndsvda"

    #sp_mat=sp.csr_matrix(mat)
    

    # Fit the NMF model:frobenius
    print(
    "Fitting the NMF model (Frobenius norm) with tf-idf features, "
    "n_samples=%d and n_features=%d..." % (n_samples, n_features)
    )
    t0 = time()
    from nmf import run_nmf
    W,H,err = run_nmf(mat, n_components,use_gpu=use_gpu,beta_loss=2.0, init=init, algo=algo )#,verose=False,n_jobs
    #print(err)
    print("done in %0.3fs." % (time() - t0))
    #scale_smc_nmf.index
    return W,H
	
	
	
def plot_top_genes(H, feature_names, n_top_words, title,list):
    import matplotlib.pyplot as plt
    import scipy.sparse as sp
    fig, axes = plt.subplots(list[0],list[1], figsize=(40, 30), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(H):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Cluster {topic_idx}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
def get_top_genes(H, feature_names, n_top_words):
    import matplotlib.pyplot as plt
    import scipy.sparse as sp
    mat=[]
    for topic_idx, topic in enumerate(H):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        #print(top_features)
        #print("----------------")
        #print(weights)
        mat.append(top_features)
    #print(mat)
    return pd.DataFrame(np.array(mat).T)
    

def get_max_loading(W):
    labels_ = []
    for j in range(W.shape[0]): 
        #print(j)
    
        mapping={}  
        for i,v in enumerate(W[j]):
            mapping[v]=i
        a=mapping[max(W[j])]
        #print(a)
        labels_.append(a)
    labels=np.array(labels_) 
    len(labels)    
    labels.astype(str)
    return labels
    

	
