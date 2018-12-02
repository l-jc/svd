import numpy as np 
import datetime
import random
import os
from svd2 import SVD
from svdpp2 import SVDPP
from optimizer2 import *
import sys


def main():
    try:
        name = str(sys.argv[1])
        K = int(sys.argv[2])
        lam = float(sys.argv[3])
        epochs = int(sys.argv[4])
        if len(sys.argv) > 5:
            save = int(sys.argv[5])
        else:
            save = 0
    except:
        print("usage: python %s <model> <K> <lambda> <epochs> [save]"%sys.argv[0])
        return

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
    # K = 15

    # random.seed(100)
    # random.shuffle(data)

    half = len(data) // 2
    quarter = len(data) // 4
    train = data[:half]
    valid = data[half:half+quarter]
    test = data[half+quarter:]

    if name == 'svd':
        model = SVD(M,N,K)
    elif name == 'svd++':
        model = SVDPP(M,N,K)
    else:
        raise NotImplementedError

    if save:
        if name == 'svd':
            model.load("svd")
        elif name == 'svd++':
            model.load("svd++")

    optimizer = RMSPROP(lr=0.001)
    model.fit(train,valid,optimizer,lam=lam,epochs=epochs)
    
    if save:
        if name == 'svd':
            model.save("svd")
        elif name == 'svd++':
            model.save("svd++")

    mse = model.validate(test)
    print("test mse", mse)


if __name__ == '__main__':
    main()