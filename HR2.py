import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import time
from collections import Counter
import re, nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import folium
from matplotlib.colors import LinearSegmentedColormap
import missingno as msno

df = pd.read_pickle('After_filling_Nans')
df.Hotel_Name.describe()

Hotel_Name_count = df.Hotel_Name.value_counts()
Hotel_Name_count[:10].plot(kind='bar',figsize=(10,8))

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 50, 18
rcParams["axes.labelsize"] = 16
from matplotlib import pyplot
import seaborn as sns
data_plot = df[["Hotel_Name","Average_Score"]].drop_duplicates()
sns.set(font_scale = 2.5)
a4_dims = (30, 12)
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.countplot(ax = ax,x = "Average_Score",data=data_plot)

text = ""
for i in range(df.shape[0]):
    text = " ".join([text,df["Reviewer_Nationality"].values[i]])

from wordcloud import WordCloud
wordcloud = WordCloud(background_color='black', width = 600,\
                      height=200, max_font_size=50, max_words=40).generate(text)
wordcloud.recolor(random_state=312)
plt.imshow(wordcloud)
plt.title("Wordcloud for countries")
plt.axis("off")
plt.show()

df.Reviewer_Nationality.describe()
Reviewer_Nat_Count = df.Reviewer_Nationality.value_counts()
print(Reviewer_Nat_Count[:10])

df.Review_Date.describe()

Review_Date_count = df.Review_Date.value_counts()
Review_Date_count[:10].plot(kind='bar')

Reviewers_freq = df.Total_Number_of_Reviews_Reviewer_Has_Given.value_counts()
Reviewers_freq[:10].plot(kind='bar')

Reviewers_freq[:10]

temp_df = df.drop_duplicates(['Hotel_Name'])
len(temp_df)

map_osm = folium.Map(location=[47, 6], zoom_start=5, tiles = 'Stamen Toner' )

temp_df.apply(lambda row:folium.Marker(location=[row["lat"], row["lng"]]).add_to(map_osm), axis=1)

print(map_osm)

pos_words = df.Review_Total_Positive_Word_Counts.value_counts()
print(pos_words[:10])

a = df.loc[df.Review_Total_Positive_Word_Counts == 0]
print('No of completely Negative reviews in the dataset:',len(a))
b = a[['Positive_Review','Negative_Review']]
print(b[:10])

neg_words = df.Review_Total_Negative_Word_Counts.value_counts()
print(neg_words[:10])

a = df.loc[df.Review_Total_Negative_Word_Counts == 0 ]
print('No of completely positive reviews in the dataset:',len(a))
b = a[['Positive_Review','Negative_Review']]
print(b[:10])

df['pos_count']=0
df['neg_count']=0

df['Negative_Review']=[x.lower().strip() for x in df['Negative_Review']]
df['Positive_Review']=[x.lower().strip() for x in df['Positive_Review']]

df["neg_count"] = df.apply(lambda x: 1 
    if x["Positive_Review"] == 'no positive' 
    or x['Positive_Review']=='nothing' 
    or x['Negative_Review']=='everything'
    else x['pos_count'],axis = 1)

df["pos_count"] = df.apply(lambda x: 1 
    if x["Negative_Review"] == 'no negative' 
    or x['Negative_Review']=='nothing' 
    or x['Positive_Review']=='everything'
    else x['pos_count'],axis = 1)

df.pos_count.value_counts()

df.neg_count.value_counts()

reviews = pd.DataFrame(df.groupby(["Hotel_Name"])["pos_count","neg_count"].sum())

print(reviews.head())

reviews["HoteL_Name"] = reviews.index
reviews.index = range(reviews.shape[0])
reviews.head()

reviews["total"] = reviews["pos_count"] + reviews["neg_count"]
reviews["pos_ratio"] = reviews["pos_count"].astype("float")/reviews["total"].astype("float")

famous_hotels = reviews.sort_values(by = "total",ascending=False).head(100)
pd.set_option('display.max_colwidth', 2000)
popular = famous_hotels["HoteL_Name"].values[:20]
popular_hotels =df.loc[df['Hotel_Name'].isin(popular)][[
    "Hotel_Name","Hotel_Address",'Average_Score','lat','lng'
    ]].drop_duplicates()
maps_osm = folium.Map(location=[47, 6], 
    zoom_start=5, tiles = 'Stamen Toner' )
popular_hotels.apply(lambda row:folium.Marker(
    location=[row["lat"], row["lng"]]).add_to(maps_osm), axis=1)

print(maps_osm)

print(popular_hotels)

pos = famous_hotels.sort_values(by = "pos_ratio",ascending=False)["HoteL_Name"].head(20).values
famous_pos = df.loc[df['Hotel_Name'].isin(pos)][[
    "Hotel_Name","Hotel_Address",'lat','lng','Average_Score']].drop_duplicates()
positive_map = folium.Map(location=[47, 6], zoom_start=5, tiles = 'Stamen Toner' )
famous_pos.apply(lambda row:folium.Marker(location=[row["lat"], row["lng"]]).add_to(positive_map), axis=1)

print(positive_map)

print(famous_pos)

reviews.to_pickle('reviews')




