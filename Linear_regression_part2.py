#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pandas_profiling import ProfileReport
import numpy as np


# In[4]:


df = pd.read_csv('Advertising.csv')


# In[5]:


df


# In[8]:


df.head(10)


# In[7]:


df.tail()


# In[9]:


df.describe()


# In[10]:


ProfileReport(df)


# In[11]:


pf = ProfileReport(df)


# In[13]:


pf.to_widgets()


# In[14]:


df


# In[15]:


pf.to_file('test.html')


# In[16]:


df


# In[26]:


x = df[["TV"]]
x


# In[27]:


y = df.sales


# In[28]:


y


# In[29]:


from sklearn.linear_model import LinearRegression
linear = LinearRegression()


# In[31]:


linear.fit(x,y)


# In[32]:


linear.intercept_


# In[33]:


linear.coef_


# In[34]:


file = 'linear_reg.sav'
pickle.dump(linear,open(file,'wb'))


# In[38]:


linear.predict([[45]])


# In[39]:


l = [4,5,6,7,89,34,45,67,23]


# In[42]:


for i in l :
    print(linear.predict([[i]]))


# In[43]:


m = 0
b = 0

learning_rate = 0.001
for i in range(len(x)):
    x1 = x.iloc[0,:].values[0]
    y1 = y.iloc[0,:].values[0]
    guess = m * x1 + b
    error = y1 - guess
    m -= error * x1 * learning_rate
    b -= error * learning_rate
    print(error)


# In[44]:


def gd(x,y,n):
    m=6
    c=3
    alpha=0.8
    len=x.shape[0]
    for r in range(n):
        y_pred=m*x +c
        cost=(1/len)*((y_pred-y)**2).sum()
        print('cost is ',cost)
        m=m-(-alpha/len)*((y-y_pred)*x).sum()
        c=c-(-alpha/len)*((y-y_pred).sum())
        print('m,c is',m,c)
        if cost<1:
            plt.scatter(x,y)
            plt.plot(x,y_pred)


# In[ ]:


file = 'linear_reg.sav'
pickle.dump(linear,open(file,'wb'))


# In[45]:


saved_model = pickle.load(open(file,'rb'))


# In[46]:


saved_model.predict([[45]])


# In[47]:


linear.score(x,y)


# In[ ]:




