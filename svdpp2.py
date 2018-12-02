import numpy as np 
import datetime
from optimizer2 import *
import os

# print("svdpp v2 imported")

class SVDPP():
    def __init__(self,M,N,K):
        """ 
        M: number of users; 
        N: number of items; 
        K: number of dimensions
        """
        self.M = M
        self.N = N
        self.K = K
        self.A = 0
        self.Bi = np.random.rand(self.M).astype('float32')
        self.Bj = np.random.rand(self.N).astype('float32')
        self.U = np.random.rand(self.M,self.K).astype('float32')
        self.V = np.random.rand(self.K,self.N).astype('float32')
        self.A = 5 - np.mean(self.Bi[0] + self.Bj + np.dot(self.U[0],self.V))
        print("model svd++ | M:",self.M,"N:",self.N,"K:",self.K, "A: ",self.A)


    def fit(self,data,valid,optimizer,lam=1.0,epochs=10):
        print("training | lambda:",lam,end=' ')
        print("optimizer:",optimizer)
        if isinstance(optimizer,RMSPROP):
            optimizer.initialize([(1,),self.Bi.shape,self.Bj.shape,self.U.shape, self.V.shape])

        for e in range(epochs):
            dU = np.array([ [0.0] * self.K ] * self.M).astype('float32')
            dV = np.array([ [0.0] * self.N ] * self.K).astype('float32')
            dBi= np.array([ 0.0 ] * self.M).astype('float32')
            dBj= np.array([ 0.0 ] * self.N).astype('float32')
            dA = 0
            for i,j,r in data:
                r_ = self.A + self.Bi[i] + self.Bj[j] + np.dot(self.U[i,...], self.V[...,j]) - r
                dA += r_
                dBi[i] += r_
                dBj[j] += r_
                dU[i,...] += (np.dot(self.U[i,...], self.V[...,j]) - r) * self.V[...,j]
                dV[...,j] += (np.dot(self.U[i,...], self.V[...,j]) - r) * self.U[i,...]
            dBi+= lam * self.Bi
            dBj+= lam * self.Bj
            dU += lam * self.U
            dV += lam * self.V
            optimizer.optimize([dA,dBi,dBj,dU,dV],[self.A,self.Bi,self.Bj,self.U,self.V])

            # self.A -= lr * 0.001 * dA
            # self.Bi-= lr * dBi
            # self.Bj-= lr * dBj
            # self.U -= lr * dU
            # self.V -= lr * dV

            train_mse = self.validate(data)
            valid_mse = self.validate(valid)
            print("epoch:",e+1,"/",epochs,"train mse:",train_mse,"valid mse:",valid_mse)

    def validate(self,data):
        se = 0
        for i,j,r in data:
            se += (self.A + self.Bi[i] + self.Bj[j] + np.dot(self.U[i,...], self.V[...,j]) - r) ** 2
        return se / len(data)

    def predict(self,i,j):
        return self.A + self.Bi[i] + self.Bj[j] + np.dot(self.U[i,...], self.V[...,j])


    def save(self,path):
        if os.path.exists(path):
            os.mkdir(path)
        open(os.path.join(path,"A++"),"w").write(repr(self.A))
        np.save(os.path.join(path,"Bi++"),self.Bi)
        np.save(os.path.join(path,"Bj++"),self.Bj)
        np.save(os.path.join(path,"U++"),self.U)
        np.save(os.path.join(path,"V++"),self.V)
        print("model saved")

    def load(self,path):
        try:
            self.A = eval(open(os.path.join(path,"A++")).read())
            self.Bi= np.load(os.path.join(path,"Bi++.npy"))
            self.Bj= np.load(os.path.join(path,"Bj++.npy"))
            self.U = np.load(os.path.join(path,"U++.npy"))
            self.V = np.load(os.path.join(path,"V++.npy"))
            print("load model")
        except:
            pass


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

    half = len(data) // 2
    quarter = len(data) // 4
    train = data[:half]
    valid = data[half:half+quarter]
    test = data[half+quarter:]

    model = SVDPP(M,N,K)

    # mse = model.validate(data)
    # print("mse on train:",mse)

    # model.load()
    optimizer = RMSPROP()
    model.fit(train,valid,optimizer,lam=6,epochs=10)
    # model.save()

    mse = model.validate(data)
    print("mse on train:",mse)


if __name__ == '__main__':
    main()