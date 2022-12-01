# INTERNSHIP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("online_shoppers_intention.csv")

df.shape

df.info()

df.head()

df.tail()

df['SpecialDay'].value_counts()

df.corr()

df.describe()

df.isnull().sum()

df['Weekend'].value_counts()

df['Revenue'].value_counts()

df.skew()

df.kurt()

df_num = df.drop(['Month','Weekend', 'Revenue'], axis = 1)
fig, ax = plt.subplots(nrows = 2, ncols = 7, figsize = (15,8))

for variable, subplot in zip(df_num.columns, ax.flatten()):
    sns.boxplot(df[variable], ax = subplot)
plt.show()

## remove outliers

Q1 = df_num.quantile(0.25)
Q3 = df_num.quantile(0.75)

IQR = Q3 - Q1

df_num = df_num[~((df_num<(Q1-1.5*IQR))|(df_num>(Q3+1.5*IQR))).any(axis=1)]
df_num.shape

# checking outlier using boxplot

df_num = df.drop(['Month','Weekend', 'Revenue'], axis = 1)
fig, ax = plt.subplots(nrows = 2, ncols = 7, figsize = (15,8))

for variable, subplot in zip(df_num.columns, ax.flatten()):
    sns.boxplot(df_num[variable], ax = subplot)
plt.show()

## Visualization 

df.columns

sns.set_theme(style='darkgrid')
sns.countplot(df['Weekend'], palette='autumn')
plt.show()

print("From this plot we can say very less people visited the shopping website in the weekend")

df['OperatingSystems'].value_counts()

a1=[2,1,3,4,8,6,7,5]
a2=[6601,2585,2555,478,79,19,7,6]

import plotly.express as px
import plotly.offline as py
import plotly.graph_objs as go

fig = go.Figure(data=[go.Pie(labels=a1, values=a2, title='Percentage of Different OS used by visitors')])

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


print("Most visitors use OS of type 2, as we can observe from the pie chart")

df['Browser'].value_counts()

b1=[2,1,4,5,6,10,8,3,13,7,12,11,9]
b2 = [7961,2462,736,467,174,163,135,105,61,49,10,6,1]

fig = go.Figure(data=[go.Pie(labels=b1, values=b2, title='Percentage of Different Browser used by visitors')])

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


print("-->  most of the visitors use Browser of typ 2")

df['Region'].value_counts()

c1 = [1,3,4,2,6,7,9,8,5]
c2 = [4780,2403,1182,1136,805,761,511,434,318]

fig = go.Figure(data=[go.Pie(labels=c1, values=c2, title='Percentage of Different Region used by visitors')])

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


print("-->  most of the visitors are from region 1")

df['TrafficType'].value_counts()

d1 = [2,1,3,4,13,10,6,8,5,11,20,9,7,15,19,14,18,16,12,17]
d2 = [3913,2451,2052,1069,738,450,444,343,260,247,198,42,40,38,17,13,10,3,1,1]

fig = go.Figure(data=[go.Pie(labels=d1, values=d2, title='Percentage of Different Traffic type used by visitors')])

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


print("--> most visitors are from traffic type 2")


sns.set(rc={'figure.figsize': (15,8)})
sns.set_theme(style = 'white')
sns.countplot(df['Month'], hue=df['VisitorType'], palette = 'deep')
plt.legend(bbox_to_anchor = (1.05, 1))
plt.show()

print("1.  Return visitor from month may")
print("2.  New visitors from november")

sns.set(rc={'figure.figsize': (15,8)})
sns.set_theme(style = 'white')
sns.countplot(df['Month'], hue=df['SpecialDay'], palette = 'deep')
plt.legend(bbox_to_anchor = (1, 0.6))
plt.show()

print("1. There are no special days between Aug and Nov")
print("2. Many special days in month of may")
print("3. very few special day in the month of Oct")

## Distribution plots

sns.set(rc={'figure.figsize':(15,8)})
sns.set_theme(style='white')
sns.distplot(df['Administrative'])

sns.set(rc={'figure.figsize':(15,8)})
sns.set_theme(style='white')
sns.distplot(df['Administrative_Duration'])

sns.set(rc={'figure.figsize':(15,8)})
sns.set_theme(style='white')
sns.distplot(df['Informational'])

sns.set(rc={'figure.figsize':(15,8)})
sns.set_theme(style='white')
sns.distplot(df['Informational_Duration'])

sns.set(rc={'figure.figsize':(15,8)})
sns.set_theme(style='white')
sns.distplot(df['ProductRelated'])

sns.set(rc={'figure.figsize':(15,8)})
sns.set_theme(style='white')
sns.distplot(df['ProductRelated_Duration'])

sns.set(rc={'figure.figsize':(15,8)})
sns.set_theme(style='white')
sns.distplot(df['BounceRates'])

sns.set(rc={'figure.figsize':(15,8)})
sns.set_theme(style='white')
sns.distplot(df['ExitRates'])

sns.set(rc={'figure.figsize':(15,8)})
sns.set_theme(style='white')
sns.distplot(df['PageValues'])

# heat map

plt.figure(figsize=(15,16))
ax = sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.show()

from sklearn import preprocessing

## Label Encoding

def Label_encoding(c1):
    label_encoder = preprocessing.LabelEncoder()
    df[c1] = label_encoder.fit_transform(df[c1])
    df[c1].unique
    return df

Label_encoding('Revenue')

df = pd.get_dummies(df, columns=['Month', 'Weekend', 'VisitorType'])

df

## Feature Engineering

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df)

feature = df.drop('Revenue', axis = 1)
label = df['Revenue']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size= 0.3)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_train.shape)



## Data Modelling

import xgboost as xgb
U_train = xgb.DMatrix(data = X_train, label = y_train)
U_test = xgb.DMatrix(data = X_test, label = y_test)

param = {'eta': 0.3, 'max_depth':3,'objective': 'multi:softprob','num_class':3}
steps = 20

model = xgb.train(param, U_train, steps)

import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

preds = model.predict(U_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))


## clustering



from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

pca = PCA(2)

df3 = pca.fit_transform(df)

df3.shape

kmeans = KMeans(3)
kmeans.fit(df3)

Sum_of_squared_distances = []
K = range(1,8)
for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(df)
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K,Sum_of_squared_distances, 'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')
plt.show()

identified_clusters = kmeans.fit_predict(df3)
identified_clusters

filter_label = df[identified_clusters==0]

plt.scatter(filter_label.iloc[:,0], filter_label.iloc[:,1])
plt.show()


# unique labels


u_labels = np.unique(identified_clusters)
ax = plt.subplot(111, projection = '3d', label='bla')

for i in u_labels:
    ax.scatter(df3[identified_clusters==i,0], df3[identified_clusters==i, 1], label=i)
plt.legend()
plt.show()

