#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('USA_Housing.csv')


# In[3]:


df


# In[7]:


df.info()


# In[8]:


df.describe()


# # Exploratory Data Analysis

# In[15]:


sns.pairplot(df)


# In[18]:


sns.distplot(df['Price'])


# In[ ]:





# In[ ]:





# In[ ]:





# # training Linear regression  model

# In[ ]:


# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets. 
# Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column. 


# In[22]:


df.columns


# # creating X and Y array

# In[25]:


x=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y=df['Price']


# # Train Test Model
# Now let's split the data into a training set and a testing set. We will train out model on the training set and then use the test set to evaluate the model

# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=101)


# In[32]:


y_test


# In[33]:


y_test.shape


# # creating and training the model

# In[38]:


import sklearn
from sklearn.linear_model import LinearRegression


# 

# In[39]:


lm=sklearn.linear_model.LinearRegression()


# In[40]:


lm.fit(x_train,y_train)


# # Model Evaluation
# Let's evaluate the model by checking out it's coefficients and how we can interpret them.

# In[41]:


x.columns


# In[43]:


lm.coef_


# In[46]:


#here we r printing the intercept
print(lm.intercept_)


# In[48]:


coeff_df=pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])


# In[49]:


coeff_df


# # Now we r making prediction for our models

# In[53]:


predicti=lm.predict(x_test)
predicti


# In[55]:


plt.scatter(y_test,predicti)


# In[57]:


sns.distplot((y_test-predicti),bins=70)


# In[58]:


from sklearn import metrics


# In[62]:


print('MAE:',metrics.mean_absolute_error(y_test,predicti))
print('MSE',metrics.mean_squared_error(y_test,predicti))
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,predicti)))


# In[ ]:




