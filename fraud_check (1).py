#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("Fraud_check.csv")
df


# In[3]:


# Exploratory Data Analysis


# In[4]:


df.head()


# In[5]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


df.info()


# In[10]:


df.describe()


# In[3]:


df=pd.get_dummies(df,columns=['Undergrad','Marital.Status','Urban'],drop_first=True)


# In[4]:


df["TaxInc"]=pd.cut(df["Taxable.Income"],bins=[10002,30000,99620],labels=["Risky","Good"])


# In[5]:


df=pd.get_dummies(df,columns=["TaxInc"],drop_first=True)


# In[6]:


df.tail(10)


# In[7]:


def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)


# In[8]:


df_norm=norm_func(df.iloc[:,1:])
df_norm.tail(10)


# In[9]:


x=df_norm.drop(['TaxInc_Good'],axis=1)
y=df_norm['TaxInc_Good']


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)


# In[12]:


from sklearn.ensemble import RandomForestClassifier


# In[14]:


forest_new=RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=20,criterion='gini')
forest_new.fit(xtrain,ytrain)


# In[17]:


forest_new.score(xtrain,ytrain)


# In[18]:


forest_new.score(xtest,ytest)


# In[ ]:




