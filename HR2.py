#!/usr/bin/env python
# coding: utf-8

# ### 4. Exploratory Data Analysis

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

import seaborn as sns

import cufflinks as cf

import warnings
warnings.filterwarnings('ignore')

import missingno as msno


# In[6]:


#Load data dari file pickle Clean_Data
data = pd.read_pickle('Clean_Data')


# #### 4.1 Hotel_Name

# In[7]:


data.Hotel_Name.describe()


# Terdapat __1492__ hotel dan yang paling banyak mendapat reviews adalah hotel __Britannia International Hotel Canary Wharf__ dengan total reviews sebanyak __4789__

# In[9]:


#Mengidentifikasi 10 teratas hotel dengan reviews terbanyak
Hotel_Name_count = data.Hotel_Name.value_counts()
Hotel_Name_count[:10].plot(kind='bar',figsize=(10,8))


# #### 4.2 Average_Score

# In[10]:


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 50, 18
rcParams["axes.labelsize"] = 16


# In[12]:


avg_plot = data[["Hotel_Name","Average_Score"]].drop_duplicates()
sns.set(font_scale = 2.5)
a4_dims = (30, 12)
fig, ax = plt.subplots(figsize=a4_dims)
sns.countplot(ax = ax,x = "Average_Score",data=avg_plot)


# __Average_Score__ hotel berada pada kisaran __8.0 - 9.1__

# #### 4.3 Reviewer_Nationality

# In[14]:


text = ""
for i in range(data.shape[0]):
    text = " ".join([text,data["Reviewer_Nationality"].values[i]])


# In[15]:


from wordcloud import WordCloud
wordcloud = WordCloud(background_color='white', width = 600,                      height=200, max_font_size=50, max_words=40).generate(text)
wordcloud.recolor(random_state=312)
plt.imshow(wordcloud)
plt.title("Wordcloud for countries ")
plt.axis("off")
plt.show()


# In[16]:


data.Reviewer_Nationality.describe()


# Terdapat __227 Kewarganegaraan yang berbeda__. 
# __United Kingdom__ menjadi __negara terbanyak__ yang memberikan review dengan __frekuensi reviews sebesar 245.110__ (__47.57%__ dari total reviews)

# In[17]:


#Mengidentifikasi 10 Teratas Kewarganegaraan yang Terbanyak Memberikan Reviews
Reviewer_Nat_Count = data.Reviewer_Nationality.value_counts()
print(Reviewer_Nat_Count[:10])


# #### 4.4 Review_Date

# In[18]:


data.Review_Date.describe()


# Terdapat 731 tanggal review yang berbeda.
# __Mayoritas reviewers memberikan review pada tanggal 8/2/2017__ dengan __frekuensi reviews__ sebesar __2584__.

# In[19]:


#Mengidentifikasi 10 Tanggal Teratas Reviews Terbanyak
Review_Date_count = data.Review_Date.value_counts()
Review_Date_count[:10].plot(kind='bar')


# #### 4.5 Total_Number_of_Reviews_Reviewer_Has_Given

# In[20]:


Reviewers_freq = data.Total_Number_of_Reviews_Reviewer_Has_Given.value_counts()
Reviewers_freq[:10].plot(kind='bar')


# In[21]:


Reviewers_freq[:10]


# Sebanyak __154506 (29.99% dari total reviews) reviewers baru pertama kali memberikan review__

# #### 4.6 Review_Total_Positive_Word_Counts

# In[22]:


pos_words = data.Review_Total_Positive_Word_Counts.value_counts()
pos_words[:10]


# In[23]:


a = data.loc[data.Review_Total_Positive_Word_Counts == 0]
print('Number of completely Negative reviews in the dataset:',len(a))
b = a[['Positive_Review','Negative_Review']]
b[:10]


# __Sebanyak 35904 reviews adalah murni negatif__

# #### 4.7 Review_Total_Negative_Word_Counts

# In[24]:


neg_words = data.Review_Total_Negative_Word_Counts.value_counts()
neg_words[:10]


# In[26]:


a = data.loc[data.Review_Total_Negative_Word_Counts == 0 ]
print('No of completely positive reviews in the dataset:',len(a))
b = a[['Positive_Review','Negative_Review']]
b[:10]


# __Sebanyak 127757 reviews adalah murni negatif__
