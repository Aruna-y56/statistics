#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv("bank data.csv")
df


# In[3]:


df.head(10)


# In[4]:


df.isnull().sum()


# In[5]:


df.describe().T


# In[6]:


No_sub=len(df[df['y']=='no'])
Sub=len(df[df['y']=='yes'])
percent_No_sub=(No_sub/len(df['y']))*100
percent_sub=(Sub/len(df['y']))*100
print('percentage of subscription:',percent_sub)
print('percentage of no subscription:',percent_No_sub)


# In[7]:


df['y'].value_counts().plot.bar()


# In[8]:


for col in df:
    pd.crosstab(df[col],df.y).plot(kind='bar')
    plt.title(col)


# In[10]:


sns.countplot(x="job",data=df,palette="hls")


# In[11]:


pd.crosstab(df.head().job,df.age).plot(kind="bar")


# In[12]:


pd.crosstab(df.head().marital,df.balance).plot(kind="bar")


# In[13]:


pd.crosstab(df.head().education,df.balance).plot(kind="bar")


# In[14]:


df.isnull().sum()


# In[15]:


df.shape


# In[16]:


df.dropna()


# In[17]:


df.shape


# In[15]:


#I dont have any null values


# In[18]:


df['pdays_no_contact']=(df['pdays']==-1)*1
contact=({'cellular':0,'telephone':1})
df['contact']=df['contact'].map(contact)


# In[19]:


data=pd.get_dummies(df,columns=['job','marital','education','default','housing','loan','month','poutcome'],drop_first=True)
data


# In[20]:


data.shape


# In[23]:


from sklearn.model_selection import train_test_split
df.shape
X = data.loc[:,df.columns!='y']
Y = data.loc[:,df.columns=='y']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


# In[22]:


print('Length of X_train : ',len(X_train),'\n length of Y_train :',len(Y_train))
print('\n Lenth of X_test : ',len(X_test), '\n length of Y_test  :',len(Y_test))


# In[141]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# In[153]:


import pandas as pd
def clean_dataset(df):
    assert isinstance(df,pd.DataFrame),"df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep=~df.isin([np.nan,np.inf,-np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


# In[155]:


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,Y_train)
print('Train Accuracy :',clf.score(X_train,Y_train))
print('Test Accuracy :',clf.score(X_test,Y_test))


# In[ ]:





# In[ ]:





# In[ ]:




