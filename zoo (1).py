#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


# In[27]:


df_animals=pd.read_csv("Zoo.csv")


# In[28]:


df_animals.head()


# In[29]:



df_animals.plot(kind="bar")


# In[30]:


plt.figure(figsize=(10,8))
df_animals.domestic.value_counts().plot(kind="bar")
plt.xlabel("Is Domestic")
plt.ylabel("Count")
plt.plot()


# In[31]:


df_animals.milk.value_counts()


# In[32]:


df_animals[(df_animals.milk==1)].shape[0]


# In[33]:



df_animals.aquatic.value_counts()


# In[34]:



df_animals[df_animals.aquatic==1]


# In[35]:


df_animals.venomous.value_counts()


# In[36]:


x=df_animals.loc[:,["milk","backbone","toothed","venomous","domestic","aquatic"]].values
y=df_animals.iloc[:,17].values


# In[37]:


y.shape


# In[38]:


x.shape


# In[39]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[40]:



from sklearn.neighbors import KNeighborsClassifier


# In[41]:



clf=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)


# In[42]:


clf.fit(x_train,y_train)


# In[43]:


y_pred=clf.predict(x_test)
y_pred


# In[44]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score


# In[45]:



accuracy_score(y_test,y_pred)


# In[46]:


from sklearn.linear_model import LogisticRegression
clf_log=LogisticRegression(random_state=0)


# In[47]:


clf_log.fit(x_train,y_train)


# In[48]:


y_pred


# In[49]:



from sklearn.metrics import confusion_matrix,accuracy_score,precision_score


# In[50]:


accuracy_score(y_test,y_pred)


# In[51]:


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


# In[ ]:




