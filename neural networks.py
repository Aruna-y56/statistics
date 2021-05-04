#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy
import pandas
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error


# In[9]:


get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install ipykernel')


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.constraints import maxnorm


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# In[13]:


seed=7
numpy.random.seed(seed)


# In[ ]:


# Exploratory data analysis


# In[16]:


dataframe=pandas.read_csv("forestfires.csv")


# In[ ]:


dataframe.month.replace('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True
dataframe.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7),inplace=True)


# In[ ]:


dataframe.head() # head will going to give you top 5 rows


# In[ ]:


dataframe.describe()


# In[ ]:


dataframe.shape()


# In[ ]:


dataframe.dtypes


# In[ ]:


dataframe.isnull.sum()


# In[ ]:


dataframe.corr(method='pearson')


# In[ ]:


dataset=dataframe.values
X=dataset[:,0:12]
Y=dataset[:,12]


# In[ ]:


model=ExtraTreesRegressor()
rfe=RFE(model,3)
fit=rfe.fit(X,Y)
fit.n_features_
fit.support_
fit.ranking_


# In[17]:


plt.hist((dataframe.area))


# In[18]:


dataframe.hist()


# In[26]:


dataframe.plot(kind='density',subplots=True,sharex=False,sharey=False)


# In[27]:


scatter_matrix(dataframe)


# In[28]:


fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(dataframe.corr(),
vmin=-1,vmax=1)
fig.colorbar(cax)
ticks=numpy.arange(0,13,1)
ax.set_xtickets(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(dataframe.columns)
ax.set_yticklabels(dataframe.columns)


# In[ ]:


num_instances = len(X)

models = []
models.append(('LiR', LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('Lasso', Lasso()))
models.append(('ElasticNet', ElasticNet()))
models.append(('Bag_Re', BaggingRegressor()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVM', SVR()))


# In[ ]:


results = []
names = []
scoring = []


# In[ ]:


for name, model in models:
    
    model.fit(X, Y)
    
    predictions = model.predict(X)
    
    
    score = explained_variance_score(Y, predictions)
    mae = mean_absolute_error(predictions, Y)
    
    results.append(mae)
    names.append(name)
    
    msg = "%s: %f (%f)" % (name, score, mae)
    print(msg)


# In[ ]:


Y = numpy.array(Y).reshape((len(Y), 1))
scaler = MinMaxScaler(feature_range=(0, 1))
Y = scaler.fit_transform(Y)


# In[ ]:


def baseline_model():
    model = Sequential()
    model.add(Dense(10, input_dim=12, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(3, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
seed = 7
numpy.random.seed(seed)


estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=600, batch_size=5, verbose=0)

kfold = KFold(n_splits=30, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

