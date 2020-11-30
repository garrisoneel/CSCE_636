import torch
from helper import load_data
from solution import PCA, AE, frobeniu_norm_error
import numpy as np
import os


def test_pca(A, p):
    pca = PCA(A, p)
    Ap, G = pca.get_reduced()
    A_re = pca.reconstruction(Ap)
    error = frobeniu_norm_error(A, A_re)
    print('PCA-Reconstruction error for {k} components is'.format(k=p), error)
    return G

def test_ae(A, p):
    model = AE(d_hidden_rep=p)
    model.train(A, A, 128, 300)
    A_re = model.reconstruction(A)
    final_w = model.get_params()
    error = frobeniu_norm_error(A, A_re)
    print('AE-Reconstruction error for {k}-dimensional hidden representation is'.format(k=p), error)
    return final_w

if __name__ == '__main__':
    dataloc = "./data/USPS.mat"
    A = load_data(dataloc)
    A = A.T
    ## Normalize A
    A = A/A.max()

    ### YOUR CODE HERE
    n = A.shape[1]
    A_norm = (A - ((1/n)*A@np.ones((n,1)))@np.ones((1,n)))
    ps = [32,64,128]
    # PCA VS AE (shared weights)
    for p in ps:
        G = test_pca(A_norm, p)
        final_w = test_ae(A, p)

    #compare G and final_w
    # G:256x64
    R = G.T@W
    p=64
    G = test_pca(A_norm,p)
    W = test_ae(A_norm,p)
     
    R = G.T@W
    u,s,v = np.linalg.svd(R,full_matrices=True)
    Rp = u@np.eye(u.shape[0],v.shape[0])@v
    Gp = G@Rp
    print("Gp-W", frobeniu_norm_error(Gp,W))
    print('R-Rp', frobeniu_norm_error(R,Rp))
    A_re = Gp@Gp.T@A_norm
    print("Gp reconstruction error:", frobeniu_norm_error(A_norm,A_re))


    #AE (shared) vs AE(non-shared)
    for p in ps:
        final_w = test_ae(A,p)

    #multilayer AE
    ### END YOUR CODE 
