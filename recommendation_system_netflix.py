#!/usr/bin/env python
# coding: utf-8

# ### Load Libraries

# In[2]:


import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from datetime import datetime
from multiprocessing import Pool
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
import h5py
np.random.seed(7)


# ### Defining Defaults and Common Functions

# In[3]:


n_max_samples = 1000000
n_custs   = 30000
datadir = './datafiles/' 


# In[4]:


def saveh5py(hdata,hname):
    h5f = h5py.File(datadir+hname, 'w')
    h5f.create_dataset('dataset', data=hdata)
    h5f.close()

def readh5py(hname):
    h5f = h5py.File(datadir+hname,'r')
    hdata = h5f['dataset'][:]
    h5f.close()
    return hdata


# ### Load Data from raw Netflix Prize file(s)

# In[4]:


def read_data(load_saved=False):
    initTime = datetime.now()
    if load_saved:
        data = (np.genfromtxt(datadir+'current_full_data.csv',dtype=int))
        print ("Time spent in data load: "+str(datetime.now() - initTime))
        return data
    data = []
    for filename in ['1']: #['1','2','3','4']:
        filename = '../dataset/combined_data_'+filename+'.txt'
        for line in open(filename,'r').read().strip().split('\n'):
            line = line.strip()
            if line.endswith(':'):
                movie_title = (line[:-1])
            else:
                data.append([movie_title]+line.split(','))            
    data = (np.array(data)[:,:-1]).astype(int)
    title = 'movie,username,rating,timestamp'
    np.savetxt(datadir+'current_full_data.csv',data,fmt="%i",header=title)
    print ("Time spent in data load: "+str(datetime.now() - initTime))
    return data


# ### Data sampling and cleaning

# In[5]:


# Random Sampling 
def sample_data(data):
    return data[np.random.choice(data.shape[0], n_max_samples, replace=False),:]


# In[6]:


# To get indices of randomly selected movies and customers and if needed, restrict their numbers
def get_indices(data):
    cust_ids = np.unique(data[:,1])
    cust_ids = cust_ids[np.random.choice(len(cust_ids), n_custs, replace=False)]
    cust_ids.sort()
    cust_ids = list(cust_ids)
    print('final number of customers: '+str(len(cust_ids)))
    
    movie_ids = np.unique(data[:,0])
    movie_ids.sort()
    movie_ids = list(movie_ids)
    print('final number of movies: '+str(len(movie_ids)))
    return cust_ids,movie_ids


# ## Data Load and process

# #### Run only for a fresh start

# In[7]:


print ("Calling function to load data")
tdata = read_data(load_saved=True)
print ("Loaded full data")


# #### Run only for a fresh start

# In[8]:


xtrain = sample_data(tdata)
print ("Sampled the data to include {} datapoints".format(n_max_samples))
global_var_cust_ids, global_var_movie_ids = get_indices(xtrain)
print ("Listed the IDs of coustomers and movies as global variables")
np.savetxt(datadir+'final_custids.csv',global_var_cust_ids,fmt="%i")
np.savetxt(datadir+'final_movieids.csv',global_var_movie_ids,fmt="%i")
print ("Saved the global variables to disk")


# ### Convert Movie and User Indices to continuous

# In[9]:


def convert_indices(item):
    if item[1] in global_var_cust_ids:
        item[1] = global_var_cust_ids.index(item[1])+1
        item[0] = global_var_movie_ids.index(item[0])+1
    else:
        item = [0,0,0]
    return item


# #### Run only for a fresh start

# In[10]:


print ("User IDs are discontinuous in this dataset. Converting indices to continuous.")
initTime = datetime.now()
p = Pool()
xtrain = p.map(convert_indices,xtrain)
print ("Time spent on conversion: "+str(datetime.now() - initTime))
xtrain = np.array(xtrain)
xtrain = xtrain[~np.all(xtrain == 0, axis=1)]
print ("Shape of final full data after conversion: "+str(xtrain.shape))
title = 'movie,username,rating'
np.savetxt(datadir+'converted_final_data.csv',xtrain,fmt="%i",header=title)


# #### Run only for a fresh start

# In[11]:


print ("Splitting data to train and test sets")
xtrain, xtest = train_test_split(xtrain, test_size=0.2, random_state=7)
saveh5py(xtrain,'traindata.h5')
saveh5py(xtest,'testdata.h5')
print ("Shape of training data: "+str(xtrain.shape))
print ("Shape of test data: "+str(xtest.shape))


# ## Compute User-Item Matrix

# In[12]:


def gen_dmatrix(data):
    dmatrix = np.zeros((len(global_var_cust_ids),len(global_var_movie_ids)))
    for item in data:
        dmatrix[item[1]-1][item[0]-1] = item[2]
    saveh5py(dmatrix,'dmatrix.h5')
    return dmatrix


# #### Run only for a fresh start

# In[13]:


initTime = datetime.now()
dmatrix = gen_dmatrix(xtrain)
print ("Time spent on computing data matrix: "+str(datetime.now() - initTime))
print ("Shape of user-item matrix: "+str(dmatrix.shape))
print ("Number of users: {} , Number of Items: {}".format(len(global_var_cust_ids),len(global_var_movie_ids)))


# ## Compute User-Item and Similarity Matrices

# #### Run only for a fresh start

# In[14]:


comp_user_similarity = True
comp_movie_similarity = True
sim_metric = 'cosine'


# #### Run only for a fresh start

# In[15]:


if comp_user_similarity:
    initTime = datetime.now()
    user_similarity = pairwise_distances(dmatrix, metric=sim_metric)
    print (user_similarity.shape)
    print ("Time spent on user similarity matrix: "+str(datetime.now() - initTime))
    saveh5py(user_similarity,'usersim.h5')


# #### Run only for a fresh start

# In[16]:


if comp_movie_similarity:
    initTime = datetime.now()
    item_similarity = pairwise_distances(dmatrix.T, metric=sim_metric)
    print ("Time spent on item similarity matrix: "+str(datetime.now() - initTime))
    print (item_similarity.shape)
    saveh5py(item_similarity,'itemsim.h5')


# In[31]:


# This is the only one snippet that I fully and blindly copied from the analyticsvidhya script. Yet to look into its details.
def user_predict(ratings, similarity):
    print ("Beginning User Predict")
    initTime = datetime.now()
    mean_user_rating = ratings.mean(axis=1)
    #We use np.newaxis so that mean_user_rating has same format as ratings
    ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
    print ("Proceeding to prediction")
    pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    print ("Time spent on computing prediction matrix: "+str(datetime.now() - initTime))
    return pred


# In[25]:


def movie_predict(ratings, similarity):
    pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


# #### Run only for a fresh start

# In[27]:


predictions_user = user_predict(dmatrix,user_similarity)
saveh5py(predictions_user,'userpreds.h5')


# #### Run only for a fresh start

# In[28]:


predictions_movie = movie_predict(dmatrix,item_similarity)
saveh5py(predictions_movie,'itempreds.h5')


# #### Run for a restart

# In[22]:


restarting_from_computed = False
if restarting_from_computed:
    dmatrix = readh5py('dmatrix.h5')
    user_similarity      = readh5py('usersim.h5')
    item_similarity      = readh5py('itemsim.h5')
    predictions_movie    = readh5py('userpreds.h5')
    predictions_user     = readh5py('itempreds.h5')
    xtrain               = readh5py('traindata.h5')
    xtest                = readh5py('testdata.h5')
    global_var_cust_ids  = np.genfromtxt(datadir+'final_custids.csv',dtype=int)
    global_var_movie_ids = np.genfromtxt(datadir+'final_movieids.csv',dtype=int)


# In[30]:


#max([max(item) for item in predictions_user])


# In[29]:


#max([max(item) for item in dmatrix])

