import numpy as np
import pickle
import os

def loadCifar(folder="./cifar-10", mode="onehot"):
    X_train = None
    y_train = None
    for i in range(1, 6):
        fname = "data_batch_{}".format(i)
        fold = pickle.load(open(os.path.join(folder, fname), 'rb'), encoding='latin1')
        if X_train is None:
            X_train = fold['data']
            y_train = fold['labels']
        else:
            X_train = np.vstack((X_train, fold['data']))
            y_train.extend(fold['labels'])
            
    np.random.seed(42)
    subIndex = np.arange(0, len(X_train))
    np.random.shuffle(subIndex)
    subIndexValid = subIndex[:10000]
    subIndexTrain = subIndex[10000:]
    np.random.seed()

    #subindex = np.arange(0, len(X_train)) # np.random.randint(0, len(X_train), (5000,))
    y_train = np.array(y_train)
    X_train = X_train.reshape(-1, 3, 32, 32) / np.float32(256)
    
    fold = pickle.load(open(os.path.join(folder, "test_batch"), 'rb'), encoding='latin1')
    X_test = fold['data']
    X_test = X_test.reshape(-1, 3, 32, 32) / np.float32(256)
    
    y_test = np.array(fold['labels'])
    
    y_train_mat = np.zeros((y_train.shape[0], 10), dtype=np.uint8)
    y_train_mat[np.arange(len(y_train)), y_train] = 1
    
    y_test_mat = np.zeros((y_test.shape[0], 10), dtype=np.uint8)
    y_test_mat[np.arange(len(y_test)), y_test] = 1
    
    if mode == 'onehot':
        return X_train[subIndexTrain], y_train_mat[subIndexTrain], X_train[subIndexValid], y_train_mat[subIndexValid], X_test, y_test_mat
    else:
        assert False, "Unsupported!"
        return X_train, y_train, X_test, y_test
