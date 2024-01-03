import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import pandas as pd
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import spsolve
from pandas.api.types import CategoricalDtype
import random
import implicit
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.datasets.movielens import get_movielens
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight, tfidf_weight)
import tqdm
import codecs

"""
# Recommend for a customer on graciousstyle
"""

retail_data = pd.read_json('data/cus_user_quality_date_0716.json') # This may take a couple minutes
# df2 = pd.read_csv('data/product_brand_type_color.csv') # This may take a couple minutes

cleaned_retail = retail_data.loc[pd.isnull(retail_data.PRODUCT_NAME) == False]
cleaned_retail = cleaned_retail[cleaned_retail['PRODUCT_DIM_KEY'].isin(cleaned_retail[cleaned_retail['ORDER_DATE_KEY'] > 17533].PRODUCT_DIM_KEY)]
cleaned_retail['QUANTITY'] = cleaned_retail['QUANTITY'].fillna(0.0)
cleaned_retail = cleaned_retail[cleaned_retail['RESOLUTION_STATUS'] == 'COMPLETED']
print('Duplicated rows: ' + str(cleaned_retail.duplicated().sum()))
cleaned_retail.drop_duplicates(inplace=True)

def threshold_likes(df, uid_min, mid_min, user_column = 'CUSTOMER_DIM_KEY', item_column = 'PRODUCT_DIM_KEY'):
    n_users = df[user_column].unique().shape[0]
    n_items = df[item_column].unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_users*n_items) * 100
    print('Starting likes info')
    print('Number of users: {}'.format(n_users))
    print('Number of models: {}'.format(n_items))
    print('Sparsity: {:4.3f}%'.format(sparsity))
    
    done = False
    while not done:
        starting_shape = df.shape[0]
        mid_counts = df.groupby(user_column)[item_column].count()
        df = df[~df[user_column].isin(mid_counts[mid_counts < mid_min].index.tolist())]
        uid_counts = df.groupby(item_column)[user_column].count()
        df = df[~df[item_column].isin(uid_counts[uid_counts < uid_min].index.tolist())]
        ending_shape = df.shape[0]
        if starting_shape == ending_shape:
            done = True
    
    assert(df.groupby(user_column)[item_column].count().min() >= mid_min)
    assert(df.groupby(item_column)[user_column].count().min() >= uid_min)
    
    n_users = df[user_column].unique().shape[0]
    n_items = df[item_column].unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_users*n_items) * 100
    print('Ending likes info')
    print('Number of users: {}'.format(n_users))
    print('Number of models: {}'.format(n_items))
    print('Sparsity: {:4.3f}%'.format(sparsity))
    return df

df_lim = threshold_likes(cleaned_retail, 2, 2)
item_lookup = df_lim[['PRODUCT_DIM_KEY', 'PRODUCT_NAME']].drop_duplicates() # Only get unique item/description pairs
#item_lookup['PRODUCT_DIM_KEY'] = item_lookup.PRODUCT_DIM_KEY.astype(str) # Encode as strings for future lookup ease

def create_matrix(cleaned_retail, row_feature, col_feature):
    cleaned_retail[row_feature] = cleaned_retail[row_feature].astype(int) # Convert to int for customer ID
    cleaned_retail = cleaned_retail[[col_feature, 'QUANTITY', row_feature]] # Get rid of unnecessary info
    grouped_cleaned = cleaned_retail.groupby([row_feature, col_feature]).sum().reset_index() # Group together
    grouped_cleaned.QUANTITY.loc[grouped_cleaned.QUANTITY == 0] = 1 # Replace a sum of zero purchases with a one to
    # indicate purchased
    grouped_purchased = grouped_cleaned.query('QUANTITY > 0') # Only get customers where purchase totals were positive
    
    customers = list(np.sort(grouped_purchased[row_feature].unique())) # Get our unique customers
    products = list(grouped_purchased[col_feature].unique()) # Get our unique products that were purchased
    quantity = list(grouped_purchased.QUANTITY) # All of our purchases

    rows = grouped_purchased[row_feature].astype('category', CategoricalDtype(categories = customers)).cat.codes 
    # Get the associated row indices
    cols = grouped_purchased[col_feature].astype('category', CategoricalDtype(categories = products)).cat.codes 
    # Get the associated column indices
    purchases_sparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(customers), len(products)))
    return purchases_sparse, customers, products

purchases_sparse, products, customers = create_matrix(df_lim, 'PRODUCT_DIM_KEY', 'CUSTOMER_DIM_KEY')

customers_arr = np.array(customers) # Array of customer IDs from the ratings matrix
products_arr = np.array(products) # Array of product IDs from the ratings matrix

def make_train(ratings, pct_test = 0.2):
    '''
    This function will take in the original user-item matrix and "mask" a percentage of the original ratings where a
    user-item interaction has taken place for use as a test set. The test set will contain all of the original ratings, 
    while the training set replaces the specified percentage of them with a zero in the original ratings matrix. 
    
    parameters: 
    
    ratings - the original ratings matrix from which you want to generate a train/test set. Test is just a complete
    copy of the original set. This is in the form of a sparse csr_matrix. 
    
    pct_test - The percentage of user-item interactions where an interaction took place that you want to mask in the 
    training set for later comparison to the test set, which contains all of the original ratings. 
    
    returns:
    
    training_set - The altered version of the original data with a certain percentage of the user-item pairs 
    that originally had interaction set back to zero.
    
    test_set - A copy of the original ratings matrix, unaltered, so it can be used to see how the rank order 
    compares with the actual interactions.
    
    user_inds - From the randomly selected user-item indices, which user rows were altered in the training data.
    This will be necessary later when evaluating the performance via AUC.
    '''
    test_set = ratings.copy() # Make a copy of the original set to be the test set. 
    test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
    training_set = ratings.copy() # Make a copy of the original data we can alter as our training set. 
    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of user,item index into list
    random.seed(0) # Set the random seed to zero for reproducibility
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement
    user_inds = [index[0] for index in samples] # Get the user row indices
    item_inds = [index[1] for index in samples] # Get the item column indices
    training_set[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
    return training_set, test_set, list(set(user_inds)) # Output the unique list of user rows that were altered

product_train, product_test, product_users_altered = make_train(purchases_sparse, pct_test = 0.2)

model = AlternatingLeastSquares(factors=80,regularization=0.0001)
# The literature suggests values between 0 and 3 for K1 and 0 and 1 for B. 
ratings = bm25_weight(product_train, K1= 1, B=0.9).tocsr()
model.fit(ratings)

def get_rec_items(model, customer_id, sparse_matrix, customer_list, item_list, item_lookup, num_items = 10):
    cust_ind = np.where(customer_list == customer_id)[0][0] # Returns the index row of our customer id
    purchased_ind = sparse_matrix[:,cust_ind].nonzero()[0] # Get column indices of purchased items
    prod_codes = item_list[purchased_ind] # Get the stock codes for our purchased items
    print(prod_codes)
    print(item_lookup.loc[item_lookup.PRODUCT_DIM_KEY.isin(prod_codes)])
    
    recom = model.recommend(cust_ind, sparse_matrix.T.tocsr(),filter_items=purchased_ind.tolist(), N= num_items)
    rec_keys = [] # start empty list to store items
    descriptions = []
    for index in recom:
        key = item_list[index[0]]
        rec_keys.append(key)
        descriptions.append(item_lookup.PRODUCT_NAME.loc[item_lookup.PRODUCT_DIM_KEY == key].iloc[0])
        
    scores = [item[1] for item in recom]
    final_frame = pd.DataFrame({'PRODUCT_DIM_KEY': rec_keys, 'PRODUCT_NAME': descriptions, 'SCORE': scores}) # Create a dataframe 
    return final_frame[['PRODUCT_DIM_KEY', 'PRODUCT_NAME','SCORE']] # Switch order of columns around

get_rec_items(model, 81494, product_train, customers_arr, products_arr, item_lookup, num_items = 20)