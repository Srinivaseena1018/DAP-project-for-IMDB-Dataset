#!/usr/bin/env python
# coding: utf-8

# In[ ]:


IMPORTING THE DATASET


# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv("movie_metadata.csv")
df


# In[ ]:


DISPLAYING THE TOP 5 ROWS IN THE DATASET


# In[4]:


df.head()


# In[ ]:


DISPLAYING THE LAST 5 ROWS IN THE DATASET


# In[5]:


df.tail()


# In[ ]:


DISPLAYING THE 10 SAMPLE ROWS IN THE DATASET


# In[6]:


df.sample(10)


# In[ ]:


Info of Data Set


# In[7]:


df.info()


# In[ ]:


Shape of the DataSet


# In[10]:


df.shape


# In[ ]:


DISPLAYING THE DESCRIPTION AND SUMMARY OF THE DATA


# In[11]:


df.info()
print("------------------------")
df.shape


# In[ ]:


DISPLAYING THE COLUMN HEADINGS


# In[13]:


df.columns


# In[ ]:


DATA PREPARATION
1.CHECKING FOR DUPLICATE DATA


# In[14]:


df.duplicated()


# In[ ]:


2.CHECKING FOR MISSING DATA


# In[15]:


df.isnull().sum()


# In[ ]:


DATA CONTAINS SOME VALUES SO WE DROP THOSES ROWS


# In[16]:


df.dropna()


# In[ ]:


DELETING THE DUPLICATE VALUES IN THE DATA


# In[17]:


df=df.drop_duplicates()
df


# In[ ]:


REVOMING UNWANTED COLUMNS


# In[34]:


df.drop(['actor_2_facebook_likes','cast_total_facebook_likes','facenumber_in_poster','movie_facebook_likes','color','actor_1_facebook_likes','actor_3_facebook_likes','director_facebook_likes'], inplace=True, axis=1)
print(df)


# In[ ]:


DISPLAYING THE STATISTICAL INFORMATION OF THE DATA


# In[35]:


df.describe()


# In[33]:


print('Out of all % of records not commited fraud==> ',len(df[df['imdb_score']==0])/len(df)*100)
print('Out of all % of records commited fraud==> ',len(df[df['duration']==1])/len(df)*100)


# In[27]:


from sklearn.utils import resample
oversamp=resample(df[df['FraudFound_P']==1],
 replace=True,
 n_samples=len(df[df['FraudFound_P']==0]),
 random_state=30)
df1=pd.concat([oversamp, df[df['FraudFound_P']==0]])


# In[28]:


df1.shape


# In[29]:


df1.head()


# In[39]:


plt.figure(figsize=(8,4))
sns.histplot(data = df, x = 'imdb_score', kde = True,bins=20)
plt.title('IMdb scores of the movie')
plt.show()


# In[43]:


categorical_features = df.columns[(df.dtypes == 'object') == True].to_list()

numerical_df = df.drop(categorical_features, axis=1)
categorical_df = df[categorical_features]


# In[44]:


categorical_df.head(),numerical_df.head()


# In[46]:


sns.pairplot(numerical_df)


# In[ ]:


IMDB score depending on movie title


# In[1]:


plt.figure(figsize=(15,5))
sns.scatterplot(x=categorical_df['movie_title'], y=numerical_df['imdb_score'])
plt.xticks(rotation=70, horizontalalignment='right', 
           fontsize=9)

print()


# In[ ]:


Relationship Betweeen imdb_score & aspect_ratio


# In[52]:


new_df = df[df['imdb_score'] != 0]

plt.figure(figsize=(12,6))
sns.scatterplot(data = new_df,x = 'imdb_score', y = 'aspect_ratio')
plt.xlabel('IMDB Scores')
plt.ylabel("Aspect Ratio")
plt.title("IMDB Scores VS Aspect Ratio")
plt.show()


# In[69]:


plt.figure(figsize=(10,5))
sns.scatterplot(x=categorical_df['language'], y=numerical_df['title_year'])
plt.xticks(rotation=40, horizontalalignment='right', 
           fontsize=9)

print()

