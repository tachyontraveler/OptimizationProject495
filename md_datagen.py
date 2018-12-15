#!/usr/bin/env python
# coding: utf-8

# ## A script to sample data from the huge Netflix prize dataset so as to validate the ML algorithms within limited computational power

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import h5py
import sys
np.random.seed(7)


# In[ ]:


n_max_samples = 3000000 ## Maximum number of samples in the dataset
n_custs   = 50000       ## Number of customers (users) to be included in the datset
datadir = './datafiles/'   ## Directory to save and retrieve data
load_saved = False      ## To get the initial full raw data from file current_full_data.h5


# In[ ]:


def OWrite(s):
    print(s)
    sys.stdout.flush()
    
def saveh5py(hdata,hname):
    h5f = h5py.File(datadir+hname, 'w')
    h5f.create_dataset('dataset', data=hdata)
    h5f.close()

def readh5py(hname):
    h5f = h5py.File(datadir+hname,'r')
    hdata = h5f['dataset'][:]
    h5f.close()
    return hdata


# In[ ]:


## To read raw data
def read_data():
    initTime = datetime.now()
    if load_saved:
        data = readh5py('current_full_data.h5')
        OWrite ("Time spent in data load: "+str(datetime.now() - initTime))
        return data
    data = []
    for filename in ['1']: #['1','2','3','4']:
        filename = datadir+'dataset/combined_data_'+filename+'.txt'
        for line in open(filename,'r').read().strip().split('\n'):
            line = line.strip()
            if line.endswith(':'):
                movie_title = (line[:-1])
            else:
                data.append([movie_title]+line.split(','))            
    data = (np.array(data)[:,:-1]).astype(int)
    data[:,[0,1]] = data[:,[1,0]]
    saveh5py(data,'current_full_data.h5')
    OWrite ("Time spent in data load: "+str(datetime.now() - initTime))
    return data


# Random sampling
def sample_data(data):
    return data[np.random.choice(data.shape[0], n_max_samples, replace=False),:]


# To get indices of randomly selected movies and customers and if needed, restrict their numbers
def get_indices(data):
    cust_ids,n_counts = np.unique(data[:,0],return_counts=True)
    cust_ids = cust_ids[np.argpartition(n_counts, -1*n_custs)[-1*n_custs:]]
                #cust_ids[np.random.choice(len(cust_ids), n_custs, replace=False)]
    cust_ids.sort()
    cust_ids = list(cust_ids)
    OWrite('final number of customers: '+str(len(cust_ids)))
    
    movie_ids = np.unique(data[:,1])
    movie_ids.sort()
    movie_ids = list(movie_ids)
    OWrite('final number of movies: '+str(len(movie_ids)))
    return cust_ids,movie_ids


# In[ ]:


# Data Load and process

OWrite ("Calling function to load the full raw data")
xtrain = read_data()
OWrite ("Loaded full data")


# In[ ]:


# To limit the dataset size so that the algorithms can be run in the local systems
xtrain = sample_data(xtrain)
OWrite ("Sampled the data to include {} datapoints".format(n_max_samples))
global_var_cust_ids, global_var_movie_ids = get_indices(xtrain)
OWrite ("Listed the IDs of coustomers and movies as global variables")
np.savetxt(datadir+'final_custids.csv',global_var_cust_ids,fmt="%i")
np.savetxt(datadir+'final_movieids.csv',global_var_movie_ids,fmt="%i")
OWrite ("Saved the global variables to disk")


# In[ ]:


# Convert Movie and User Indices to continuous

def convert_indices(item):
    if item[0] in global_var_cust_ids:
        item[0] = global_var_cust_ids.index(item[0])
        item[1] = global_var_movie_ids.index(item[1])
    else:
        item = [0,0,0]
    return item


# In[ ]:


## A parallelized system to convert non-continuous Netflix user IDs to continuous for convenience
OWrite ("User IDs are discontinuous in this dataset. Converting indices to continuous.")
initTime = datetime.now()
p = Pool()
xtrain = p.map(convert_indices,xtrain)
OWrite ("Time spent on conversion: "+str(datetime.now() - initTime))
xtrain = np.array(xtrain)
xtrain = xtrain[~np.all(xtrain == 0, axis=1)]
OWrite ("Shape of final full data after conversion: "+str(xtrain.shape))
saveh5py(xtrain,'converted_final_data.h5')

