#!/usr/bin/env python
# coding: utf-8

# ## References
# 
# http://kitchingroup.cheme.cmu.edu/blog/2017/11/18/Neural-networks-for-regression-with-autograd/
# 
# https://github.com/HIPS/autograd/blob/master/autograd/misc/optimizers.py
# 
# https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/
# 
# https://medium.com/datadriveninvestor/how-to-built-a-recommender-system-rs-616c988d64b2

# ### Load Libraries

# In[ ]:


from autograd import numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import h5py
import sys
import autograd.numpy.random as npr
import pickle
np.random.seed(7)


# ### Defining Defaults and Common Functions

# In[ ]:


datadir = './datafiles/'


# In[ ]:


def OWrite(s):
    print(s)
    sys.stdout.flush()
    
def saveh5py(hdata,hname):
    with h5py.File(datadir+hname, 'w') as h5f:
        h5f.create_dataset('dataset', data=hdata)

def readh5py(hname):
    with h5py.File(datadir+hname,'r') as h5f:
        hdata = h5f['dataset'][:]
    return hdata

def gen_dmatrix(data,matrix_shape):
    initTime = datetime.now()
    dmatrix = np.zeros(matrix_shape)
    for item in data:
        dmatrix[item[0],item[1]] = item[2]
    saveh5py(dmatrix,'dmatrix.h5')
    OWrite ("Time spent on computing data matrix: "+str(datetime.now() - initTime))
    return dmatrix


# In[ ]:


class RecSys():
    def __init__(self,args):
        
        self.L           = args['L']
        self.alpha       = args['alpha']
        self.xtrain      = args['xtrain']
        self.dmatrix     = args['dmatrix']
        self.num_users   = len(args['unique_cust_ids_list'])
        self.num_movies  = len(args['unique_movie_ids_list'])
        self.max_epochs  = args['max_epochs']
        self.gradient_fn = grad(self.squared_error,0)
        
        self.UL = np.random.normal(scale=1./self.L, size=(self.num_users, self.L))
        self.ML = np.random.normal(scale=1./self.L, size=(self.num_movies, self.L))
        self.BU = np.zeros((self.num_users,1))
        self.BM = np.zeros((self.num_movies,1))
        self.b  = np.mean(xtrain[:,-1])
        

    def model(self,params):
        ul,ml,bu,bm = params
        return self.b + bu + bm + np.dot(ul,ml.T)

    def squared_error(self,params,y):
        return (np.square(y-self.model(params)))
    
    def gradient_descent(self):
        for xtem in self.xtrain:
            params = (self.UL[xtem[0]],self.ML[xtem[1]],self.BU[xtem[0]],self.BM[xtem[1]])
            step_val = tuple(self.alpha*stem for stem in self.gradient_fn(params,xtem[2]))
            self.UL[xtem[0]] = params[0] - step_val[0]
            self.ML[xtem[1]] = params[1] - step_val[1]
            self.BU[xtem[0]] = params[2] - step_val[2]
            self.BM[xtem[1]] = params[3] - step_val[3]
            #params = tuple(params[i_p] - step_val[i_p] for i_p in range(len(params)))
        
    def train(self):
        lossdata = []
        for i_e in range(self.max_epochs):
            initTime = datetime.now()
            np.random.shuffle(self.xtrain)
            self.gradient_descent()
            #OWrite('Time for looping through dataset: {}'.format((datetime.now() - initTime).total_seconds()))
            avg_error = self.overall_mse()
            lossdata.append(avg_error)
            OWrite("Epoch: {} \t MSE: {:.4f} \t TimeConsumed: {}".format(
                                i_e+1,avg_error,(datetime.now() - initTime).total_seconds()))
        return lossdata
            
            
    def predict_matrix(self):
        return self.b + (recc.BU[:,np.newaxis]+recc.BM[np.newaxis:,]).squeeze() + np.dot(self.UL,self.ML.T)
    
    def overall_mse(self):
        xs, ys = self.dmatrix.nonzero()
        pred_matrix = self.predict_matrix()
        return np.sqrt(sum([pow(self.dmatrix[x,y]-pred_matrix[x,y],2) for x,y in zip(xs,ys)]))


# In[ ]:


def read_processed_data(test_size=0.1):
    data           = readh5py('converted_final_data.h5')
    list_cust_ids  = np.genfromtxt(datadir+'final_custids.csv',dtype=int)
    list_movie_ids = np.genfromtxt(datadir+'final_movieids.csv',dtype=int)
    OWrite ("Splitting data to train and test sets")
    xtrain, xtest  = train_test_split(data, test_size=test_size, random_state=7)
    dmatrix        = gen_dmatrix(xtrain,(len(global_var_cust_ids),len(global_var_movie_ids)))
    
    saveh5py(xtrain, 'traindata.h5')
    saveh5py(xtest,  'testdata.h5')
    saveh5py(dmatrix,'dmatrix.h5')
    
    OWrite ("Shape of training data: "+str(xtrain.shape))
    OWrite ("Shape of test data: "+str(xtest.shape))
    
    return (xtrain,xtest,dmatrix,list_cust_ids,list_movie_ids)


# In[ ]:


xtrain,xtest,dmatrix,list_cust_ids,list_movie_ids = read_processed_data()


# In[ ]:


args = {'alpha'     : 0.001,
        'L'         : 20,
        'xtrain'    : xtrain,
        'dmatrix'   : dmatrix,
        'max_epochs': 10,
        'unique_cust_ids_list' : np.copy(global_var_cust_ids),
        'unique_movie_ids_list': np.copy(global_var_movie_ids) ,
       }


# In[ ]:


recc = RecSys(args)


# In[ ]:


# recc.alpha = 
# recc.max_epochs = 

lossvals = recc.train()


# In[ ]:


try:
    pickle.dump(recc,open(datadir+'trained_recc_MD.pkl','wb'))
except:
    OWrite('Unable to Pickle. Proceeding to save as h5')
    train_vars = [recc.UL,recc.ML,recc.BU,recc.BM,recc.b]
    
    with h5py.File(datadir+'latest_model_trained_vars.h5', 'w') as h5f:
        for i,item in enumerate(['UL','ML','BU','BM','b']):
            h5f.create_dataset('dataset'+item, data=train_vars[i])
        


# In[ ]:





# In[ ]:




