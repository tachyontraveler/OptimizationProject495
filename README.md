# Work on Netflix Prize Dataset for EECS 495 course project (fall,2018)
### By Abhijith Gopakumar and Noopur Gupta


Since the Netflix Prize dataset is huge (about one billion data points overall), we scaled it down to be able to run in local systems. 

At first, we tested the efficiency of a recommendation system based on user similarity matrices, in this new dataset. But it turned out that such an algorithm would require a bigger subset of the Netflix Prize dataset to predict with reliability.
Code is provided in: similarity_matrices_reccs.ipynb

So further, we implemented a matrix factorization algorithm to do the job. The latent vectors' size was fixed at 10 and we chose about five million data points and restricted the number of users to 50,000. These users were selected as the ones with highest number of movie ratings. After user restrictions, the dataset size was reduced to about slightly more than a million. 
Code to generate such a datasample is provided in: datagen_Netflix_data.ipynb
(To run it for a long time in the background, it was downloaded as a python script: md_datagen.py)

On this data, we ran a matrix decomposition and optimized the latent vectors. It took about 450 seconds on an average per epoch. We ran it for 20 epochs. The results are provided inside the jupyter notebook. Errors on train and test data are reducing simultaneously. 
