
### 5. The Best Hotel Recommendation


```python
%matplotlib inline
import pandas as pd
from pandas import read_csv
from pandas import datetime
import numpy as np
from matplotlib import pyplot as plt
import math

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import missingno as msno
```


```python
#Load data dari file pickle Clean_Data
data = pd.read_pickle('Clean_Data')
```

#### 5.1 The Total Number of Positive and Negative Reviews


```python
data["pos_count"] = 1
data["neg_count"] = 1
```


```python
data["pos_count"] = data.apply(lambda x: 0 if x["Positive_Review"] == 'No Positive' else x["pos_count"],axis =1)
```


```python
data["pos_count"].value_counts()
```




    1    479308
    0     35904
    Name: pos_count, dtype: int64




```python
data["neg_count"] = data.apply(lambda x: 0 if x["Negative_Review"] == 'No Negative' else x["neg_count"],axis =1)
```


```python
data["neg_count"].value_counts()
```




    1    387455
    0    127757
    Name: neg_count, dtype: int64




```python
reviews = pd.DataFrame(data.groupby(["Hotel_Name"])["pos_count","neg_count"].sum())
```


```python
reviews["Hotel_Name"] = reviews.index
reviews.index = range(reviews.shape[0])
```


```python
reviews.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pos_count</th>
      <th>neg_count</th>
      <th>Hotel_Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>150</td>
      <td>113</td>
      <td>11 Cadogan Gardens</td>
    </tr>
    <tr>
      <th>1</th>
      <td>136</td>
      <td>123</td>
      <td>1K Hotel</td>
    </tr>
    <tr>
      <th>2</th>
      <td>660</td>
      <td>459</td>
      <td>25hours Hotel beim MuseumsQuartier</td>
    </tr>
    <tr>
      <th>3</th>
      <td>103</td>
      <td>55</td>
      <td>41</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>18</td>
      <td>45 Park Lane Dorchester Collection</td>
    </tr>
  </tbody>
</table>
</div>




```python
reviews["total"] = reviews["pos_count"] + reviews["neg_count"]
```


```python
data["count"] = 1
count_review = data.groupby("Hotel_Name",as_index=False)["count"].sum()
```

#### 5.2 Famous Hotels in Europe (based in the number of reviews)


```python
reviews = pd.merge(reviews,count_review,on = "Hotel_Name",how = "left")
```


```python
reviews.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pos_count</th>
      <th>neg_count</th>
      <th>Hotel_Name</th>
      <th>total</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>150</td>
      <td>113</td>
      <td>11 Cadogan Gardens</td>
      <td>263</td>
      <td>159</td>
    </tr>
    <tr>
      <th>1</th>
      <td>136</td>
      <td>123</td>
      <td>1K Hotel</td>
      <td>259</td>
      <td>148</td>
    </tr>
    <tr>
      <th>2</th>
      <td>660</td>
      <td>459</td>
      <td>25hours Hotel beim MuseumsQuartier</td>
      <td>1119</td>
      <td>689</td>
    </tr>
    <tr>
      <th>3</th>
      <td>103</td>
      <td>55</td>
      <td>41</td>
      <td>158</td>
      <td>103</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>18</td>
      <td>45 Park Lane Dorchester Collection</td>
      <td>46</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i in reviews.sort_values(by = "count",ascending=False)["Hotel_Name"].head(10).values:
    print(i)
```

    Britannia International Hotel Canary Wharf
    Strand Palace Hotel
    Park Plaza Westminster Bridge London
    Copthorne Tara Hotel London Kensington
    DoubleTree by Hilton Hotel London Tower of London
    Grand Royale London Hyde Park
    Holiday Inn London Kensington
    Hilton London Metropole
    Millennium Gloucester Hotel London
    Intercontinental London The O2
    


```python
reviews["pos_ratio"] = reviews["pos_count"].astype("float")/reviews["total"].astype("float")
```

#### 5.3 The Positive Hotels Among The Famous Hotels (based on the number of positive reviews of the famous hotels)


```python
famous_hotels = reviews.sort_values(by = "count",ascending=False).head(100)
```


```python
pd.set_option('display.max_colwidth', 2000)
popular = famous_hotels["Hotel_Name"].values[:10]
data.loc[data['Hotel_Name'].isin(popular)][["Hotel_Name","Hotel_Address"]].drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hotel_Name</th>
      <th>Hotel_Address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8301</th>
      <td>Grand Royale London Hyde Park</td>
      <td>1 Inverness Terrace Westminster Borough London W2 3JP United Kingdom</td>
    </tr>
    <tr>
      <th>14829</th>
      <td>Intercontinental London The O2</td>
      <td>1 Waterview Drive Greenwich London SE10 0TW United Kingdom</td>
    </tr>
    <tr>
      <th>63942</th>
      <td>Britannia International Hotel Canary Wharf</td>
      <td>163 Marsh Wall Docklands Tower Hamlets London E14 9SJ United Kingdom</td>
    </tr>
    <tr>
      <th>111930</th>
      <td>Hilton London Metropole</td>
      <td>225 Edgware Road Westminster Borough London W2 1JU United Kingdom</td>
    </tr>
    <tr>
      <th>164259</th>
      <td>Strand Palace Hotel</td>
      <td>372 Strand Westminster Borough London WC2R 0JJ United Kingdom</td>
    </tr>
    <tr>
      <th>171770</th>
      <td>Millennium Gloucester Hotel London</td>
      <td>4 18 Harrington Gardens Kensington and Chelsea London SW7 4LH United Kingdom</td>
    </tr>
    <tr>
      <th>236055</th>
      <td>DoubleTree by Hilton Hotel London Tower of London</td>
      <td>7 Pepys Street City of London London EC3N 4AF United Kingdom</td>
    </tr>
    <tr>
      <th>440985</th>
      <td>Copthorne Tara Hotel London Kensington</td>
      <td>Scarsdale Place Kensington Kensington and Chelsea London W8 5SY United Kingdom</td>
    </tr>
    <tr>
      <th>504027</th>
      <td>Park Plaza Westminster Bridge London</td>
      <td>Westminster Bridge Road Lambeth London SE1 7UT United Kingdom</td>
    </tr>
    <tr>
      <th>512645</th>
      <td>Holiday Inn London Kensington</td>
      <td>Wrights Lane Kensington and Chelsea London W8 5SP United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i in famous_hotels.sort_values(by = "pos_ratio",ascending=False)["Hotel_Name"].head(10):
    print(i)
```

    Hotel Berna
    Club Quarters Hotel Lincoln s Inn Fields
    Apex Temple Court Hotel
    Apex City Of London Hotel
    Hotel Esther a
    Urban Lodge Hotel
    Hilton London Canary Wharf
    The Piccadilly London West End
    Shangri La Hotel at The Shard London
    The Student Hotel Amsterdam City
    


```python
pos = famous_hotels.sort_values(by = "pos_ratio",ascending=False)["Hotel_Name"].head(10).values
data.loc[data['Hotel_Name'].isin(pos)][["Hotel_Name","Hotel_Address"]].drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hotel_Name</th>
      <th>Hotel_Address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>971</th>
      <td>Apex Temple Court Hotel</td>
      <td>1 2 Serjeant s Inn Fleet Street City of London London EC4Y 1LL United Kingdom</td>
    </tr>
    <tr>
      <th>147504</th>
      <td>Shangri La Hotel at The Shard London</td>
      <td>31 St Thomas Street Southwark London SE1 9QU United Kingdom</td>
    </tr>
    <tr>
      <th>223782</th>
      <td>Club Quarters Hotel Lincoln s Inn Fields</td>
      <td>61 Lincoln s Inn Fields Camden London WC2A 3JW United Kingdom</td>
    </tr>
    <tr>
      <th>228012</th>
      <td>The Piccadilly London West End</td>
      <td>65 73 Shaftesbury Avenue Westminster Borough London W1D 6EX United Kingdom</td>
    </tr>
    <tr>
      <th>273892</th>
      <td>Urban Lodge Hotel</td>
      <td>Arlandaweg 10 Westpoort 1043 EW Amsterdam Netherlands</td>
    </tr>
    <tr>
      <th>387956</th>
      <td>Apex City Of London Hotel</td>
      <td>No 1 Seething Lane City of London London EC3N 4AX United Kingdom</td>
    </tr>
    <tr>
      <th>450210</th>
      <td>Hotel Esther a</td>
      <td>Singel 303 309 Amsterdam City Center 1012 WJ Amsterdam Netherlands</td>
    </tr>
    <tr>
      <th>454064</th>
      <td>Hilton London Canary Wharf</td>
      <td>South Quay Marsh Wall Tower Hamlets London E14 9SH United Kingdom</td>
    </tr>
    <tr>
      <th>483570</th>
      <td>Hotel Berna</td>
      <td>Via Napo Torriani 18 Central Station 20124 Milan Italy</td>
    </tr>
    <tr>
      <th>509232</th>
      <td>The Student Hotel Amsterdam City</td>
      <td>Wibautstraat 129 Oost 1091 GL Amsterdam Netherlands</td>
    </tr>
  </tbody>
</table>
</div>



#### 5.4 The Most Consistent Performance Hotels


```python
data.Review_Date = pd.to_datetime(data.Review_Date)
```


```python
temp = data.groupby("Hotel_Name", as_index=False)["Reviewer_Score"].agg([np.mean, np.std]).sort_values("mean",ascending=False)
temp = temp[temp["mean"] > 8.9]
temp.shape
temp.sort_values("std").index[0:20]
```




    Index(['H10 Casa Mimosa 4 Sup', 'Hotel Casa Camper',
           'H tel de La Tamise Esprit de France', 'Le Narcisse Blanc Spa',
           'Hotel Eiffel Blomet', '45 Park Lane Dorchester Collection', '41',
           'Hotel Stendhal Place Vend me Paris MGallery by Sofitel',
           'H tel D Aubusson', 'Hotel The Serras', 'Hotel Am Stephansplatz',
           'Lansbury Heritage Hotel', 'Covent Garden Hotel', 'The Soho Hotel',
           'Catalonia Magdalenes', 'H tel Saint Paul Rive Gauche',
           'Milestone Hotel Kensington', 'Ritz Paris', 'H tel Fabric',
           'Le 123 S bastopol Astotel'],
          dtype='object', name='Hotel_Name')




```python
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
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hotel_Name</th>
      <th>Hotel_Address</th>
      <th>Average_Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11259</th>
      <td>Milestone Hotel Kensington</td>
      <td>1 Kensington Court Kensington and Chelsea London W8 5DL United Kingdom</td>
      <td>9.5</td>
    </tr>
    <tr>
      <th>20472</th>
      <td>Covent Garden Hotel</td>
      <td>10 Monmouth Street Camden London WC2H 9HB United Kingdom</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>37337</th>
      <td>Lansbury Heritage Hotel</td>
      <td>117 Poplar High Street Tower Hamlets London E14 0AE United Kingdom</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>41518</th>
      <td>Le 123 S bastopol Astotel</td>
      <td>123 boulevard S bastopol 2nd arr 75002 Paris France</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>54717</th>
      <td>Ritz Paris</td>
      <td>15 Place Vend me 1st arr 75001 Paris France</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>81101</th>
      <td>Le Narcisse Blanc Spa</td>
      <td>19 Boulevard De La Tour Maubourg 7th arr 75007 Paris France</td>
      <td>9.5</td>
    </tr>
    <tr>
      <th>111027</th>
      <td>Hotel Stendhal Place Vend me Paris MGallery by Sofitel</td>
      <td>22 Rue Danielle Casanova 1st arr 75002 Paris France</td>
      <td>8.8</td>
    </tr>
    <tr>
      <th>147195</th>
      <td>H tel Fabric</td>
      <td>31 rue de la Folie M ricourt 11th arr 75011 Paris France</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>152204</th>
      <td>H tel D Aubusson</td>
      <td>33 Rue Dauphine 6th arr 75006 Paris France</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>176748</th>
      <td>The Soho Hotel</td>
      <td>4 Richmond Mews Westminster Borough London W1D 3DH United Kingdom</td>
      <td>9.5</td>
    </tr>
    <tr>
      <th>176997</th>
      <td>H tel de La Tamise Esprit de France</td>
      <td>4 rue d Alger 1st arr 75001 Paris France</td>
      <td>9.6</td>
    </tr>
    <tr>
      <th>185602</th>
      <td>41</td>
      <td>41 Buckingham Palace Road Westminster Borough London SW1W 0PS United Kingdom</td>
      <td>9.6</td>
    </tr>
    <tr>
      <th>190062</th>
      <td>H tel Saint Paul Rive Gauche</td>
      <td>43 rue Monsieur le Prince 6th arr 75006 Paris France</td>
      <td>8.9</td>
    </tr>
    <tr>
      <th>191786</th>
      <td>45 Park Lane Dorchester Collection</td>
      <td>45 Park Lane Westminster Borough London W1K 1PN United Kingdom</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>245310</th>
      <td>Hotel Eiffel Blomet</td>
      <td>78 Rue Blomet 15th arr 75015 Paris France</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>316447</th>
      <td>Hotel Casa Camper</td>
      <td>Elisabets 11 Ciutat Vella 08001 Barcelona Spain</td>
      <td>9.6</td>
    </tr>
    <tr>
      <th>372994</th>
      <td>Catalonia Magdalenes</td>
      <td>Magdalenes 13 15 Ciutat Vella 08002 Barcelona Spain</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>398945</th>
      <td>Hotel The Serras</td>
      <td>Passeig de Colom 9 Ciutat Vella 08002 Barcelona Spain</td>
      <td>9.6</td>
    </tr>
    <tr>
      <th>402244</th>
      <td>H10 Casa Mimosa 4 Sup</td>
      <td>Pau Claris 179 Eixample 08037 Barcelona Spain</td>
      <td>9.6</td>
    </tr>
    <tr>
      <th>462731</th>
      <td>Hotel Am Stephansplatz</td>
      <td>Stephansplatz 9 01 Innere Stadt 1010 Vienna Austria</td>
      <td>9.3</td>
    </tr>
  </tbody>
</table>
</div>


