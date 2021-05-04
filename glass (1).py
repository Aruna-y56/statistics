#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# In[21]:


df=pd.read_csv("glass.csv")
df.head()


# In[22]:


x=np.array(df.iloc[:,3:5])
y=np.array(df['Type'])


# In[23]:


x.shape


# In[24]:


y.shape


# In[25]:


cm_dark = ListedColormap(['#ff6060', '#8282ff','#ffaa00','#fff244','#4df9b9','#76e8fc','#3ad628'])
cm_bright = ListedColormap(['#ffafaf', '#c6c6ff','#ffaa00','#ffe2a8','#bfffe7','#c9f7ff','#9eff93'])


# In[26]:


plt.scatter(x[:,0],x[:,1],c=y,cmap=cm_dark,s=10,label=y)
plt.show()


# In[27]:


sns.swarmplot(x='Na',y='RI',data=df,hue='Type')


# In[28]:


x_train,x_test,y_train,y_test=train_test_split(x,y)


# In[29]:


x_train.shape


# In[30]:


y_train.shape


# In[31]:


x_test.shape


# In[32]:


y_test.shape


# In[33]:


knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)


# In[34]:


pred=knn.predict(x_train)
pred


# In[35]:


accuracy=knn.score(x_train,y_train)
accuracy


# In[36]:


cnf_matrix=confusion_matrix(y_train,pred)
cnf_matrix


# In[37]:


plt.imshow(cnf_matrix,cmap=plt.cm.jet)


# In[39]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# choose k between 1 to 41
k_range = range(1, 41)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x, y, cv=5)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[40]:


df_cm=pd.DataFrame(cnf_matrix,range(6),range(6))
sns.set(font_scale=1.4)
sns.heatmap(df_cm,annot=True,annot_kws={"size":16})


# In[41]:


h = .02  
n_neighbors = 5 
for weights in ['uniform', 'distance']:
    clf = KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(x, y)
    
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  

    
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cm_bright)

    
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_dark,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"% (n_neighbors, weights))

plt.show()


# In[ ]:





# In[ ]:




