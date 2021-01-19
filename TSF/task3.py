# -*- coding: utf-8 -*-
"""Task_3.py

#*Samriddh Singh*
## **GRIP @ TSF TASK - 3**
### Problem Statement: Data visualisation to gain insights to increase the profits.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("SampleSuperstore.csv")
df = pd.DataFrame(data)
data.head(10)

"""### Understanding the data-types and unique values in each column."""

df.dtypes

data.nunique()

"""#### Since there is only one country and postal code and city are not of much importance here as we have other features to describe the same information in a better way, so we drop these columns."""

df.drop(["Postal Code","City","Country"],axis = 1, inplace =True)
df.head(10)

"""### Let us see are the numerical data related anywhere ?"""

df.corr()

"""#### A correlation of Sales vs Profit made is there which is about 0.48 .

## Data visualisation

### Let is try to plot the numerical data features against each other to visualise and find any trends if any:
"""

from pylab import rcParams
rcParams['figure.figsize'] = 8,8
X=["Sales","Quantity","Discount","Profit"]

for x in X:
  if(x!="Profit"):
   sns.lineplot(x=df[x],y=df["Profit"])
   plt.show()

"""#### Since profit and sales have slight positive correlation of about 0.48, it appears in its graph as well, though it is very vague, however since the scales of two features differ widely, let us take the logarithm of the sales feature to understand better.

#### There was a drop in the profit made when the discount was around 50%, we will find why did this occur and in which category later in our analysis.


#### Profit and quantity are also slightly correlated positively. Obviously, more the number of quantities sold, higher is the profit earned.


#### The following graph shows the relation between sales and profit
"""

fig = plt.figure()
ax = plt.gca()
ax.scatter(x=df["Sales"],y=df["Profit"], c='green',alpha = 0.3,)
ax.set_xscale('log')
plt.xlabel('Sales', fontsize=18)
plt.ylabel('Profit', fontsize=16)

"""## Let us try to understand the three segments of customers namely: Retail Consumers, Corporate Sector and Home-Office clients.

###I segregated the dataframe into these three segments and analysed the trends in the following manner:
"""

segments =["Consumer","Corporate","Home Office"]
for seg in segments:
  consumer_df = df[df.Segment == 'Consumer']
  Corporate_df = df[df.Segment == 'Corporate']
  Ho_df = df[df.Segment == 'Home Office']

consumer_df.shape

consumer_df.describe()

"""### We have about 5200 retail customers and the average profit earned fromt his segment is 25.84%

##### Let us see the following graphs to infer better.
"""

columns = ["Ship Mode","State","Category", "Sales", "Quantity", "Discount"]

for column in columns:
  fig, ax = plt.subplots()
  fig.set_size_inches(11.7, 8.27)
  sns.countplot(consumer_df[column],ax=ax)
  plt.grid()
  plt.show()
  print(consumer_df[column].value_counts())

"""## Points to note:
1. Average profit is 25.84% .
2. Majority sales are in California, New York and Texas where we can see future customers as well.
3. Consumers in the Segment purchase Office Supplies and lesser on Furniture.
4. Interstingly customers almost always prefer to buy in pairs or more: 

Quantity vs Count 


  3        1294

  2        1254

  5        642

  4        616

  1        458

"""

Corporate_df.describe()

"""### Let us analyse the segment of Corporate customers: here the average profit is about 5% higher than usual customers."""

for column in columns:
  fig, ax = plt.subplots()
  fig.set_size_inches(11.7, 8.27)
  sns.countplot(Corporate_df[column], ax=ax)
  plt.grid()
  plt.show()
  print(Corporate_df[column].value_counts())

"""## Key insights here


1. Average profit is 30% .
2. Majority sales are in California, New York and Texas where we can see future customers as well, again.
3. Consumers in the Segment purchase Office Supplies and lesser on furniture, similar to above.
4. Interstingly customers almost always prefer to buy in pairs or more:
Quantity vs Count.
"""

Ho_df.describe()

"""### The profit here is highest of all about 34%, and the deviation of profits are too high between maximum and minimum."""

for column in columns:
  fig, ax = plt.subplots()
  fig.set_size_inches(11.7, 8.27)
  sns.countplot(Ho_df[column], ax=ax)
  plt.grid()
  plt.show()
  print(Ho_df[column].value_counts())

"""The insights from this data is:

1. Average Profit is highest from this segment of branch around 34%.

2. The major customers are again from California, then it decreases significantly to half the number.

## Now let us analyse the dataset on the basis of Sub-Category column:
"""

df["Sub-Category"].value_counts()

"""#### We have 17 sub-categories in which the products are as mentioned. Now we will group them into a new dataset and find other stories learnt in selling items of each category."""

grp = df.groupby("Sub-Category").mean()
grp["Quantity"] = df["Sub-Category"].value_counts()
grp.head(17)

"""## We already have some cool insights now, let us visualise them and then we can discuss further."""

gcol = grp.columns
gcol

"""## Variation of Sales and Profit in each sub-category."""

fig, ax = plt.subplots()
fig.set_size_inches(12, 14)
sns.lineplot( x = grp.index,y = grp["Sales"], ax=ax, legend = "brief",color = "Blue")
sns.lineplot( x = grp.index,y = grp["Profit"], ax=ax, legend = "brief",color = "Green")
plt.xticks(rotation =90)
plt.grid()
plt.show()
g1 = pd.DataFrame([grp.index,grp["Profit"],grp["Sales"]])
g1.head()

"""## In the above graph note that:
#### 1. Copiers and machines have excceptionally high sales, while copier also provides the maximum profits, machines do not.
#### 2. Even though tables have good sale, they have surpirisngly given us losses.
#### 3. We can seen that the sales of Fasteners, art and labels is exceptionally low and neither are we able to generate much profit.

## Variation of Quantity and Profit in each sub-category.
"""

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
sns.lineplot( x = grp.index,y = grp["Quantity"],label ="Quantity", ax=ax,color = "Blue")
sns.lineplot( x = grp.index,y = grp["Profit"],label = "Profit", ax=ax, color = "Green")
plt.xticks(rotation =90)
plt.grid()
plt.show()
g2 = pd.DataFrame([grp.index,grp["Profit"],grp["Quantity"]])
g2.head()

"""## In the above graph note that:
#### 1. Binders and paper have maximum quantity.
#### 2. However, since we have seen that the sales of Fasteners, art and labels is exceptionally low we can reduce the quantity of them in the store.

## Variation of Discount vs Profit obtained in each sub-category.
"""

fig, ax = plt.subplots()
fig.set_size_inches(12, 14)
sns.lineplot( x = grp.index,y = 1000*grp["Discount"], ax=ax, label = "1000 x Discount",color = "Blue")
sns.lineplot( x = grp.index,y = grp["Profit"], ax=ax, label = "Profit",color = "Green")
plt.xticks(rotation =90)
plt.grid()
plt.show()
g3 = pd.DataFrame([grp.index,grp["Profit"],grp["Discount"]])
g3.head()

"""## In the above graph note that:
#### 1. Maximum discounts were provided in Binders, Machines and Tables.
#### 2. Tables have a discount of about 26% yet we are in a loss of 55%.
#### 3. Machines have discounts which may be reduced if necessary to increase profits.

### Now we can perform our analysis on how the profit and sales vary along each state in USA.
"""

sm = df.groupby("State").mean()
sm["Quantity"] = df["State"].value_counts()
sm.head(10)

"""## Based on geographical values of State we have highly varying amounts of sales, and profits !

## Instead of plotting each graph individually we will compare two trends at once against the state

## Let us plot the graphs and have a quick overview

### Profit and Sales variation
"""

fig, ax = plt.subplots()
fig.set_size_inches(20, 14)
sns.lineplot( x = sm.index,y = sm["Profit"], ax=ax, color = "Green", label ="PROFIT")
sns.lineplot( x = sm.index,y = sm["Sales"], ax=ax, color = "blue",label ="SALES")
plt.xticks(rotation =90)
plt.grid()
plt.show()
d1 = pd.DataFrame([sm.index,sm["Profit"],sm["Sales"]])
d1.head()

"""### Insights:
#### 1. Highest sales in Wyoming.
#### 2. States with good sales include Alabama, Illionis, Nevada, Rhode Island and Vermont.
#### 3. Vermont has good profits per unit sale.

### Profit and Quantity variation
"""

fig, ax = plt.subplots()
fig.set_size_inches(20, 14)
sns.lineplot( x = sm.index,y = sm["Profit"], ax=ax, legend = "brief", color = "Green", label ="PROFIT")
sns.lineplot( x = sm.index,y = sm["Quantity"], ax=ax, legend = "brief", color = "yellow", label ="QUANTITY")
plt.xticks(rotation =90)
plt.grid()
plt.show()

d2 = pd.DataFrame([sm.index,sm["Profit"],sm["Quantity"]])
d2.head()

"""### Profit and Discount variation in each state"""

fig, ax = plt.subplots()
fig.set_size_inches(20, 14)
sns.lineplot( x = sm.index,y = sm["Profit"], ax=ax, legend = "brief", color = "blue", label ="PROFIT")
sns.lineplot( x = sm.index,y = 300*sm["Discount"], ax=ax, legend = "brief", color = "green",label ="DISCOUNT x 300")
plt.xticks(rotation =90)
plt.grid()
plt.show()
d3 = pd.DataFrame([sm.index,sm["Profit"],sm["Discount"]])
d3.head()

df.head(10)

"""## Let us now group by category and look up for some trends

"""

cat = df.groupby("Category").mean()
cat["Quantity"] = df["Category"].value_counts()
cat.head(10)

fig, ax = plt.subplots()
fig.set_size_inches(20, 14)
sns.barplot( x = cat.index,y = cat["Profit"], ax=ax, label ="PROFIT")
sns.lineplot( x = cat.index, y = 100*cat["Discount"],ax = ax, label = "Discount x 100",color ="Red")
sns.lineplot( x = cat.index, y = cat["Sales"]/10,ax = ax, label = "Sales/10",color ="Yellow")
plt.grid()
plt.show()

"""### Here we can see that a whooping 78% profit was made in the technology items. 
### We can understand the red line indicating the discounts and yellow line indicating volume sales.
Note that these are not out of total sales, instead category wise percentage as provided in the dataset.

## Conclusion: Exploratory Data Analysis was completed in depth by using seaborn, matplotlib and pandas.
"""

