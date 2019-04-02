#!/usr/bin/env python
# coding: utf-8

# ### 5. The Best Hotel Recommendation

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from pandas import read_csv
from pandas import datetime
import numpy as np
from matplotlib import pyplot as plt
import math

import seaborn as sns

import cufflinks as cf

import warnings
warnings.filterwarnings('ignore')

import missingno as msno


# In[2]:


#Load data dari file pickle Clean_Data
data = pd.read_pickle('Clean_Data')


# #### 5.1 The Total Number of Positive and Negative Reviews

# In[3]:


data["pos_count"] = 1
data["neg_count"] = 1


# In[4]:


data["pos_count"] = data.apply(lambda x: 0 if x["Positive_Review"] == 'No Positive' else x["pos_count"],axis =1)


# In[5]:


data["pos_count"].value_counts()


# In[6]:


data["neg_count"] = data.apply(lambda x: 0 if x["Negative_Review"] == 'No Negative' else x["neg_count"],axis =1)


# In[7]:


data["neg_count"].value_counts()


# In[8]:


reviews = pd.DataFrame(data.groupby(["Hotel_Name"])["pos_count","neg_count"].sum())


# In[9]:


reviews["Hotel_Name"] = reviews.index
reviews.index = range(reviews.shape[0])


# In[10]:


reviews.head()


# In[11]:


reviews["total"] = reviews["pos_count"] + reviews["neg_count"]


# In[12]:


data["count"] = 1
count_review = data.groupby("Hotel_Name",as_index=False)["count"].sum()


# #### 5.2 Famous Hotels in Europe (based in the number of reviews)

# In[13]:


reviews = pd.merge(reviews,count_review,on = "Hotel_Name",how = "left")


# In[14]:


reviews.head()


# In[15]:


for i in reviews.sort_values(by = "count",ascending=False)["Hotel_Name"].head(10).values:
    print(i)


# In[16]:


reviews["pos_ratio"] = reviews["pos_count"].astype("float")/reviews["total"].astype("float")


# #### 5.3 The Positive Hotels Among The Famous Hotels (based on the number of positive reviews of the famous hotels)

# In[17]:


famous_hotels = reviews.sort_values(by = "count",ascending=False).head(100)


# In[18]:


pd.set_option('display.max_colwidth', 2000)
popular = famous_hotels["Hotel_Name"].values[:10]
data.loc[data['Hotel_Name'].isin(popular)][["Hotel_Name","Hotel_Address"]].drop_duplicates()


# In[19]:


for i in famous_hotels.sort_values(by = "pos_ratio",ascending=False)["Hotel_Name"].head(10):
    print(i)


# In[20]:


pos = famous_hotels.sort_values(by = "pos_ratio",ascending=False)["Hotel_Name"].head(10).values
data.loc[data['Hotel_Name'].isin(pos)][["Hotel_Name","Hotel_Address"]].drop_duplicates()


# #### 5.4 The Most Consistent Performance Hotels

# In[21]:


data.Review_Date = pd.to_datetime(data.Review_Date)


# In[22]:


temp = data.groupby("Hotel_Name", as_index=False)["Reviewer_Score"].agg([np.mean, np.std]).sort_values("mean",ascending=False)
temp = temp[temp["mean"] > 8.9]
temp.shape
temp.sort_values("std").index[0:20]


# In[23]:


lis = ['H10 Casa Mimosa 4 Sup', 'Hotel Casa Camper',
       'H tel de La Tamise Esprit de France', 'Le Narcisse Blanc Spa',
       'Hotel Eiffel Blomet', '45 Park Lane Dorchester Collection', '41',
       'Hotel Stendhal Place Vend me Paris MGallery by Sofitel',
       'H tel D Aubusson', 'Hotel The Serras', 'Hotel Am Stephansplatz',
       'Lansbury Heritage Hotel', 'Covent Garden Hotel', 'The Soho Hotel',
       'Catalonia Magdalenes', 'H tel Saint Paul Rive Gauche',
       'Milestone Hotel Kensington', 'Ritz Paris', 'H tel Fabric',
       'Le 123 S bastopol Astotel']
data.loc[data['Hotel_Name'].isin(lis)][["Hotel_Name","Hotel_Address","Average_Score"]].drop_duplicates()

