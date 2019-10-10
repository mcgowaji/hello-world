#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import seaborn as sns
import statsmodels.formula.api as smf
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
from sklearn import neighbors
import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import ensemble
from scipy.stats import boxcox

import warnings
warnings.filterwarnings(action='ignore')


postgres_user = 'dsbc_student'
postgres_pw = '7*.8G9QH21'
postgres_host = '142.93.121.174'
postgres_port = '5432'
postgres_db = 'fifa19'

engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(
    postgres_user, postgres_pw, postgres_host, postgres_port, postgres_db))

fifa = pd.read_sql_query('select * from fifa19',con=engine)

engine.dispose()

fifa.head(10)


# In[2]:


#Drop rows with >=10 NaNs
fifa= fifa.dropna(thresh=10)
fifa.describe()


# In[3]:


#Drop rows missing key statistics
null_rows=fifa.loc[fifa.Crossing.isnull()]
fifa=fifa.drop(null_rows.index)


# In[4]:


#Drop features that do not correlate w/ target
fifa= fifa.drop(['Contract Valid Until', 'Height', 'ID', 'Club Logo', 'Release Clause', 'Club', 'Nationality', 'Flag', 'Joined', 'Name', 'Photo', 'Real Face', 'Jersey Number', 'Position', 'Value'], axis=1)


# In[5]:


fifa=fifa.drop(['Wage'], axis=1)


# In[6]:


positions=fifa.loc[:, 'LS':'RB']
positions


# In[21]:


#Create new df using the product of each cell
summed_df = positions.applymap(lambda x: eval(x) if x else x)
summed_df= summed_df.fillna(summed_df.mean())
summed_df.shape


# In[8]:


#Convert to numeric
fifa.Weight=fifa.Weight.str.strip('lbs')
fifa.Weight= pd.to_numeric(fifa.Weight)


# In[9]:


#Defining featureset
X=fifa.drop('Overall', axis=1)


# In[10]:


#Replace positional columns with cleaned version
X=X.drop(positions, axis=1)
X=pd.concat([X, summed_df], axis=1)
X=X.fillna(X.mean())


# In[ ]:





# In[12]:


#Try Modelling with random forest regressor
X=pd.get_dummies(X)
Y=fifa['Overall']
rfr=ensemble.RandomForestRegressor(max_depth= 3, n_estimators=100)
rfr.fit(X, Y)

cross_val_score(rfr, X, Y, cv=10)


# In[23]:


X.head()


# In[35]:


#Split and analyze results
X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size= 0.20, random_state=498)

rfr=ensemble.RandomForestRegressor(max_depth= 6, n_estimators=100)

rfr.fit(X_train, y_train)

# We are making predictions here
y_preds_test = rfr.predict(X_test)

print("R-squared of the model in training set is: {}".format(rfr.score(X_train, y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(rfr.score(X_test, y_test)))
print("Mean absolute error of the prediction is: {}".format(mean_absolute_error(y_test, y_preds_test)))
print("Mean squared error of the prediction is: {}".format(mse(y_test, y_preds_test)))
print("Root mean squared error of the prediction is: {}".format(rmse(y_test, y_preds_test)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((y_test - y_preds_test) / y_test)) * 100))
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# In[36]:


#Use single decision tree for comparison
from sklearn.tree import DecisionTreeRegressor

dtr=DecisionTreeRegressor(max_depth=10)

dtr.fit(X_train, y_train)

# We are making predictions here
y_preds_test = dtr.predict(X_test)

print("R-squared of the model in training set is: {}".format(dtr.score(X_train, y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(dtr.score(X_test, y_test)))
print("Mean absolute error of the prediction is: {}".format(mean_absolute_error(y_test, y_preds_test)))
print("Mean squared error of the prediction is: {}".format(mse(y_test, y_preds_test)))
print("Root mean squared error of the prediction is: {}".format(rmse(y_test, y_preds_test)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((y_test - y_preds_test) / y_test)) * 100))

import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# Since random forest models are better at reducing _variance_ rather than bias, it is not surprising to see the decision tree outperformed the random forest on test data accuracy. It is much faster, and more easy to interpret as you may render a decision tree and see the exact criteria for the decisions made. However, due to the structure of a single decision tree, it may be prone to overfitting the noise in the dataset-since there are such low totals of samples at each leaf node 10 layers down, predictions may be completed with very few samples satisfying each and every condition met. 
# 
# Random forests will help correct this problem, as they select samples of observations of the data and random subsets of features for each tree to combat errors of variance. By reducing depth and increasing the number of trees, we may reduce both errors of variance _and_ bias to create a more predictive model that performs consistently across training and test sets. 

# In[ ]:




