import numpy as np
import torch
      
class npzloader(object):
    def __init__(self, path, device='cpu'):
        super(npzloader,self).__init__()
        self.path = path
        self.device = device
    
    def unpack(self):
        data= self.load()
        x,y = data['x'].astype(np.float32), data['y'].astype(np.float32)
        return [x,y]
        
    def load(self):
        return np.load(self.path)
    
    def toTensor(self):
        datasetList = self.unpack()
        dataset = [torch.from_numpy(item).to(self.device) for item in datasetList]
        return dataset
    
    def split(self, ntrain, nval, ntest):
        dataset = self.toTensor()

        # dataset = self.data()  #[x,y]
        samples = len(dataset[0]) #total samples in dataset
        # print(dataset[0].shape)
        # print(dataset[1].shape)

        x  = dataset[0]
        y =  dataset[1]

        xtest = x[-ntest:]
        ytest = y[-ntest:]
        ## if ntrain and ntest split 
        if ntrain + ntest + nval > samples:
            raise Exception ("Invalid ntrain, nval and ntest")
        shuffle = torch.randperm(samples-ntest)  ##  get indices to shuffle dataset
        ntrain = shuffle[:ntrain] 
        nval = shuffle[-nval:]

        xtrain, xval = x[ntrain] , x[nval] 
        ytrain, yval = y[ntrain] , y[nval]

        ## delete dataset from memory
        del x
        del y

        return [xtrain, ytrain, xval, yval, xtest, ytest]
    
    def split(self, ntrain, nval, ntest):
        dataset = self.toTensor()

        # dataset = self.data()  #[x,y]
        samples = len(dataset[0])

        x  = dataset[0]
        y =  dataset[1]

        xtest = x[-ntest:]
        ytest = y[-ntest:]
        ## if ntrain and ntest split 
        if ntrain + ntest + nval > samples:
            raise Exception ("Invalid ntrain, nval and ntest")
        shuffle = torch.randperm(samples-ntest)  ##  get indices to shuffle dataset
        ntrain = shuffle[:ntrain] 
        nval = shuffle[-nval:]

        xtrain, xval = x[ntrain] , x[nval] 
        ytrain, yval = y[ntrain] , y[nval]

        ## delete dataset from memory
        del x
        del y

        return [xtrain, ytrain, xval, yval, xtest, ytest]
    
    def cpu(self, tensor):
        return tensor.cpu()
    
    def cuda(self, tensor):
        return tensor.to(self.device)