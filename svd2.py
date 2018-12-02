import numpy as np 
import datetime
import random
import os
from optimizer2 import *

# print("svd v2 imported")

class SVD():
    def __init__(self,M,N,K):
        """ 
        M: number of users; 
        N: number of items; 
        K: number of dimensions
        """
        self.M = M
        self.N = N
        self.K = K
        self.U = np.random.rand(self.M,self.K).astype('float32')
        self.V = np.random.rand(self.K,self.N).astype('float32')
        print("model | M:",self.M,"N:",self.N,"K:",self.K)


    def fit(self,data,valid,optimizer,lam=1.0,epochs=10):
        print("training | lambda:",lam,end=' ')
        print("optimizer:",optimizer)
        if isinstance(optimizer,RMSPROP):
            print("initialize optimizer")
            optimizer.initialize([self.U.shape, self.V.shape])
        
        for e in range(epochs):
            dU = np.array([ [0.0] * self.K ] * self.M).astype('float32')
            dV = np.array([ [0.0] * self.N ] * self.K).astype('float32')
            for i,j,r in data:
                dU[i,...] += (np.dot(self.U[i,...], self.V[...,j]) - r) * self.V[...,j]
                dV[...,j] += (np.dot(self.U[i,...], self.V[...,j]) - r) * self.U[i,...]
            # dU /= len(data)
            # dV /= len(data)
            dU += lam * self.U
            dV += lam * self.V
            # optimizer api v1
            # optimizer.optimize(dU,dV,self.U,self.V)
            # optimizer api v2
            optimizer.optimize([dU,dV],[self.U,self.V])

            train_mse = self.validate(data)
            valid_mse = self.validate(valid)
            print("epoch:",e+1,"/",epochs,"train mse:",train_mse,"valid mse:",valid_mse)
            

    def validate(self,data):
        se = 0
        for i,j,r in data:
            se += (np.dot(self.U[i,...], self.V[...,j]) - r) ** 2
        return se / len(data)

    def predict(self,i,j):
        return np.dot(self.U[i,...], self.V[...,j])

    def save(self,path="svd"):
        if os.path.exists(path):
            os.mkdir(path)
        np.save(os.path.join(path,"U.npy"),self.U)
        np.save(os.path.join(path,"V.npy"),self.V)
        print("model saved")

    def load(self,path="svd"):
        up = os.path.join(path,"U.npy")
        vp = os.path.join(path,"V.npy")
        if os.path.exists(up):
            self.U = np.load(up)
            print("load U",end=' ')
        if os.path.exists(vp):
            self.V = np.load(vp)
            print("load V")
        print("")

def main():
    users = set()
    items = set()
    data = eval(open("../reviews2014").read())
    for time,user,item,rate in data:
        users.add(user)
        items.add(item)
    items = list(items)
    items.sort()
    users = list(users)
    users.sort()
    uidx = dict(zip(users,range(len(users))))
    iidx = dict(zip(items,range(len(items))))

    data = [(uidx[u],iidx[i],r) for t,u,i,r in data]
    M = len(users)
    N = len(items)
    K = 15

    model = SVD(M,N,K)

    # mse = model.validate(data)
    # print("mse on train:",mse)

    # model.load()
    optimizer = RMSPROP()
    model.fit(data,optimizer,lam=1,epochs=10)
    # model.save("svd")

    mse = model.validate(data)
    print("mse on train:",mse)


if __name__ == '__main__':
    main()