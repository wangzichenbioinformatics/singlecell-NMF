# -*- coding: utf-8 -*-
"""
Created on Fri Sep 1  2023

@author: from OU Lab. FAH,SYSU 
Wangchen wch_bioinformatics@163.com
"""

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_selection import SelectFdr, SelectPercentile, f_classif
from numpy import linalg as LA
import math
#import argparse
import itertools
import scipy.io as scio
import pandas as pd
import time
import scipy.stats as stats
from statsmodels.stats.weightstats import ttest_ind
from scipy import sparse

def coupledNMF(PeakO,P_symbol,E,E_symbol,pe):
   

    def quantileNormalize(df_input):
        df = df_input.copy()
        # compute rank
        dic = {}
        for col in df:
            dic.update({col: sorted(df[col])})
        sorted_df = pd.DataFrame(dic)
        rank = sorted_df.mean(axis=1).tolist()
        # sort
        for col in df:
            t = np.searchsorted(np.sort(df[col]), df[col])
            df[col] = [rank[i] for i in t]
        return df


    def npmax(array):
        arrayindex = array.argmax(1)
        arrayvalue = array.max(1)
        i = arrayvalue.argmax()
        j = arrayindex[i]
        return i, j



    rep = 50

    print("Loading data...")


   

    A = np.zeros((E.shape[0], PeakO.shape[0]))
    A.shape


    type(pe.values)

    for line in pe.values:
        data = line
        print(data[0],data[1],data[2],data[3])
        pindex = P_symbol.index(data[0])
        eindex = E_symbol.index(data[1])
        temp1 = float(data[3])
        if temp1 < 0:
            temp1 = 0
        temp2 = float(data[2])
        A[eindex, pindex] = math.exp(-temp2 / 30000) * temp1	


    E_symbol = np.asarray(E_symbol)
    P_symbol = np.asarray(P_symbol)
    E = pd.DataFrame(E)
    PeakO = pd.DataFrame(PeakO)
    E = quantileNormalize(E)
    PeakO = quantileNormalize(PeakO)

    K=2
    print("Initializing non-negative matrix factorization for E...")
    E[E > 10000] = 10000
    X = np.log(1 + E)

    err1 = np.zeros(rep)
    for i in range(0, rep):
        model = NMF(n_components=K, init='random', random_state=i, solver='cd', max_iter=50)
        W20 = model.fit_transform(X)
        H20 = model.components_
        err1[i] = LA.norm(X - np.dot(W20, H20), ord='fro')

    model = NMF(n_components=K, init='random', random_state=np.argmin(err1), solver='cd', max_iter=1000)
    W20 = model.fit_transform(X)
    H20 = model.components_
    S20 = np.argmax(H20, 0)

    print("Initializing non-negative matrix factorization for PeakO...")
    PeakO = np.log(PeakO + 1)
    err = np.zeros(rep)
    for i in range(0, rep):
        model = NMF(n_components=K, init='random', random_state=i, solver='cd', max_iter=50)
        W10 = model.fit_transform(PeakO)
        H10 = model.components_
        err[i] = LA.norm(PeakO - np.dot(W10, H10), ord='fro')

    model = NMF(n_components=K, init='random', random_state=np.argmin(err), solver='cd', max_iter=1000)
    W10 = model.fit_transform(PeakO)
    H10 = model.components_
    S10 = np.argmax(H10, 0)
    print("Selecting differentially expressed genes...")
    p2 = np.zeros((X.shape[0], K))
    for i in range(K):
        for j in range(X.shape[0]):
            statistic, p2[j, i], df = ttest_ind(X.iloc[j, S20 == i], X.iloc[j, S20 != i], alternative='smaller')

    WP2 = np.zeros((W20.shape))
    p2[np.isnan(p2)] = 1
    scores = -np.log10(p2)
    temp = int(len(E_symbol) / 20)
    for i in range(K):
        indexs = scores[:, i].argsort()[-temp:][::-1]
        WP2[indexs, i] = 1

    print("Selecting differentially open peaks...")
    p1 = np.zeros((PeakO.shape[0], K))
    for i in range(K):
        for j in range(PeakO.shape[0]):
            statistic, p1[j, i], df = ttest_ind(PeakO.iloc[j, S10 == i], PeakO.iloc[j, S10 != i], alternative='smaller')

    WP1 = np.zeros((W10.shape))
    p1[np.isnan(p1)] = 1
    scores = -np.log10(p1)
    temp = int(len(P_symbol) / 20)
    for i in range(K):
        indexs = scores[:, i].argsort()[-temp:][::-1]
        WP1[indexs, i] = 1

    perm = list(itertools.permutations(range(K)))
    score = np.zeros(len(perm))
    for i in range(len(perm)):
        score[i] = np.trace(np.dot(np.dot(np.transpose(WP2[:, perm[i]]), A), WP1))

    match = np.argmax(score)
    W20 = W20[:, perm[match]]
    H20 = H20[perm[match], :]
    S20 = np.argmax(H20, 0)
    print("Initializing hyperparameters lambda1, lambda2 and mu...")
    lambda10 = pow(LA.norm(X - np.dot(W20, H20), ord='fro'), 2) / pow(LA.norm(PeakO - np.dot(W10, H10), ord='fro'), 2)
    lambda20 = pow(np.trace(np.dot(np.dot(np.transpose(W20), A), W10)), 2) / pow(
        LA.norm(PeakO - np.dot(W10, H10), ord='fro'), 2)
    lambda1=0.04
    lambda2=20
    if type(lambda1) == type(None) and type(lambda2) == type(None):
        set1 = [lambda10 * pow(5, 0), lambda10 * pow(5, 1), lambda10 * pow(5, 2), lambda10 * pow(5, 3),
                lambda10 * pow(5, 4)]
        set2 = [lambda20 * pow(5, -4), lambda20 * pow(5, -3), lambda20 * pow(5, -2), lambda20 * pow(5, -1),
                lambda20 * pow(5, 0)]
    elif type(lambda1) == type(None):
        set1 = [lambda10 * pow(5, 0), lambda10 * pow(5, 1), lambda10 * pow(5, 2), lambda10 * pow(5, 3),
                lambda10 * pow(5, 4)]
        set2 = [args.lambda2]
    elif type(lambda2) == type(None):
        set1 = [lambda1]
        set2 = [lambda20 * pow(5, -4), lambda20 * pow(5, -3), lambda20 * pow(5, -2), lambda20 * pow(5, -1),
                lambda20 * pow(5, 0)]

    else:
        set1 = [lambda1 * lambda10]
        set2 = [lambda2 * lambda20]
    mu = 1
    eps = 0.001
    detr = np.zeros((len(set1), len(set2)))
    detr1 = np.zeros((len(set1), len(set2)))
    S1_all = np.zeros((len(set1) * len(set2), PeakO.shape[1]))
    S2_all = np.zeros((len(set1) * len(set2), E.shape[1]))
    P_all = np.zeros((len(set1) * len(set2), K, PeakO.shape[0]))
    E_all = np.zeros((len(set1) * len(set2), K, E.shape[0]))
    P_p_all = np.zeros((len(set1) * len(set2), K, PeakO.shape[0]))
    E_p_all = np.zeros((len(set1) * len(set2), K, E.shape[0]))
    print("Starting coupleNMF...")
    count = 0
    for x in range(len(set1)):
        for y in range(len(set2)):
            lambda1 = set1[x]
            lambda2 = set2[y]
            W1 = W10
            W2 = W20
            H1 = H10
            H2 = H20
            print(lambda1, lambda2)

            print("Iterating coupleNMF...")
            maxiter = 500
            err = 1
            terms = np.zeros(maxiter)
            it = 0
            terms[it] = lambda1 * pow(LA.norm(X - np.dot(W2, H2), ord='fro'), 2) + pow(
                LA.norm(PeakO - np.dot(W1, H1), ord='fro'), 2) + lambda2 * pow(
                np.trace(np.dot(np.dot(np.transpose(W2), A), W1)), 2) + mu * (
                                    pow(LA.norm(W1, ord='fro'), 2) + pow(LA.norm(W2, ord='fro'), 2))
            while it < maxiter - 1 and err > 1e-6:
                it = it + 1
                T1 = 0.5 * lambda2 * np.dot(np.transpose(A), W2)
                T1[T1 < 0] = 0
                W1 = W1 * np.dot(PeakO, np.transpose(H1)) / (eps + np.dot(W1, np.dot(H1, np.transpose(H1))) + 0.5 * mu * W1)
                H1 = H1 * (np.dot(np.transpose(W1), PeakO)) / (eps + np.dot(np.dot(np.transpose(W1), W1), H1))
                T2 = 0.5 * (lambda2 / lambda1 + eps) * np.dot(A, W1)
                T2[T2 < 0] = 0
                W2 = W2 * (np.dot(X, np.transpose(H2)) + T2) / (
                            eps + np.dot(W2, np.dot(H2, np.transpose(H2))) + 0.5 * mu * W2)
                H2 = H2 * (np.dot(np.transpose(W2), X) / (eps + np.dot(np.dot(np.transpose(W2), W2), H2)))
                m1 = np.zeros((K, K))
                m2 = np.zeros((K, K))
                for z in range(K):
                    m1[z, z] = LA.norm(H1[z, :])
                    m2[z, z] = LA.norm(H2[z, :])

                W2 = np.dot(W2, m2)
                W1 = np.dot(W1, m1)
                H1 = np.dot(LA.inv(m1), H1)
                H2 = np.dot(LA.inv(m2), H2)

                terms[it] = lambda1 * pow(LA.norm(X - np.dot(W2, H2), ord='fro'), 2) + pow(
                    LA.norm(PeakO - np.dot(W1, H1), ord='fro'), 2) + lambda2 * pow(
                    np.trace(np.dot(np.dot(np.transpose(W2), A), W1)), 2) + mu * (
                                        pow(LA.norm(W1, ord='fro'), 2) + pow(LA.norm(W2, ord='fro'), 2))
                err = abs(terms[it] - terms[it - 1]) / abs(terms[it - 1])

            S2 = np.argmax(H2, 0)
            S1 = np.argmax(H1, 0)

            p2 = np.zeros((X.shape[0], K))
            for i in range(K):
                for j in range(X.shape[0]):
                    statistic, p2[j, i], df = ttest_ind(X.iloc[j, S2 == i], X.iloc[j, S2 != i], alternative='smaller')

            WP2 = np.zeros((W2.shape))
            p2[np.isnan(p2)] = 1
            scores = -np.log10(p2)
            temp = int(len(E_symbol) / 20)
            for i in range(K):
                indexs = scores[:, i].argsort()[-temp:][::-1]
                WP2[indexs, i] = 1
                E_all[count, i, indexs] = 1
                E_p_all[count, i, indexs] = p2[indexs, i]

            p1 = np.zeros((PeakO.shape[0], K))
            for i in range(K):
                for j in range(PeakO.shape[0]):
                    statistic, p1[j, i], df = ttest_ind(PeakO.iloc[j, S1 == i], PeakO.iloc[j, S1 != i], alternative='smaller')

            WP1 = np.zeros((W1.shape))
            p1[np.isnan(p1)] = 1
            scores = -np.log10(p1)
            temp = int(len(P_symbol) / 20)
            for i in range(K):
                indexs = scores[:, i].argsort()[-temp:][::-1]
                WP1[indexs, i] = 1
                P_all[count, i, indexs] = 1
                P_p_all[count, i, indexs] = p1[indexs, i]

            T = np.dot(np.dot(np.transpose(WP2), A), WP1)
            temp = np.sum(np.sum(T)) * np.diag(1 / np.sum(T, axis=0)) * T * np.diag(1 / np.sum(T, axis=1))
            detr1[x, y] = np.trace(temp)
            detr[x, y] = np.trace(T)
            S1_all[count] = S1
            S2_all[count] = S2
            count = count + 1

    [i, j] = npmax(detr)
    print("Score is :", detr1[i, j] / K)
    print(
        "If the score >=1, the clustering matching for scRNA-seq and scATAC-seq is well. Otherwise, we sugguest to tune the parameters.")

    index = detr.argmax()
    S1_final = S1_all[index, :] + 1
    S2_final = S2_all[index, :] + 1
    E_final = E_all[index, :, :]
    P_final = P_all[index, :, :]
    E_p_final = E_p_all[index, :, :]
    P_p_final = P_p_all[index, :, :]
    return S1_final,S2_final