import numpy as np
import pandas as pd


arr =  np.array([1,2,3,4,5])
print(arr)
print(type(arr))


arr1 = np.array(10)
arr2 = np.array([1,2,3,4,5])
arr3 = np.array([[1,2], [3,4]])

print(arr3)
print("Dimension of arr1 is:",arr1.ndim, "\nDimension of arr2 is:",arr2.ndim,"\nDimension of arr3 is:", arr3.ndim)

arr = np.array([10,20,30,40,50])
print(arr[1:4])
print(arr[2:])
print(arr[1:4])

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(arr[0:8:2])
print(arr[0:7:3])
print(arr[1:8:2])

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[1,0:2])


arr21 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
arr_copy = arr21.copy()
arr_copy[0] =24
print(arr_copy)
print(arr21)

arr22 = np.array([[1,2,3,4], [5,6,7,8], [8,9,10,11]])
print(arr22)
print(arr22.shape)

arr23 = np.array([1,2,3,4,5,6])
print(arr23.reshape(3,2))

arr = np.array([1,2,1,4,1,6])
arrList = np.array_split(arr,3)

for arr in arrList:
    print(arr)

arr24 = np.array([1,2,1,4,5,6])
print(np.where(arr24==1))
print(np.where(arr24%2==0))
print(np.sort(arr24))

arr25 = np.array([
[[1, 2], [3, 4]],
[[5, 6], [7, 8]]
])


print(arr25.ndim)
print(arr25)

## -------------------------PANDAS--------------------------------------------------------------------------------

x= [23,48,19]
my_first_series = pd.Series(x)
print (my_first_series)


# create a dictionary

data = {
    "students": ['Emma', 'John', np.nan, 'Bob', np.nan, 'Mitsos', np.nan],
    "grades": [12,np.nan,18,17, np.nan, 15, np.nan]
}

my_first_dataframe = pd.DataFrame(data)
print(my_first_dataframe)

print(my_first_dataframe['students'])
print(my_first_dataframe['grades'])

my_first_dataframe = pd.DataFrame(data, index = ["a","b","c","d","e,","f", "g"])
first_row = my_first_dataframe.loc["a"]
print(first_row)

second_row = my_first_dataframe.iloc[1]
print(second_row)

print(my_first_dataframe.isnull())

my_first_dataframe["students"].fillna("No Name", inplace=True)
my_first_dataframe["grades"].fillna("No Grade", inplace = True)
print(my_first_dataframe)

df2 = my_first_dataframe["students"].replace(to_replace = "Bob", value = "Alice")
print(df2)


df = my_first_dataframe["grades"].interpolate(method= 'linear', limit_direction= 'forward')
print(df)

data2 = {
    "students": ['Emma', 'John', np.nan, 'Bob', np.nan, 'Mitsos', np.nan],
    "grades": [12,np.nan,18,17, np.nan, 15, np.nan]
}

df = pd.DataFrame(data2)
print(df)

df.dropna(inplace=True)
print(df)

s=pd.Series(['workearly', 'e-learning', 'python'])
for index, value in s.items():
    print(f"Index : {index}, Value: {value}")


data3 = {
    "students": ['Emma', 'John'],
     "grades": [12, 19.8]
}
df3 = pd.DataFrame(data3, index=["a", "b"])
for i, j in df3.iterrows():
      print(i, j)

#------------------- Pandas - Reading Files--------------------------------------------------------------------------

df = pd.read_csv("finance_liquor_sales.csv")
print(df.head())
print(df.info())
print(df.shape)

#------------------- Pandas - Analyzing Data--------------------------------------------------------------------------

mean = df.mean(numeric_only=True)
print(mean)
median = df.median(numeric_only=True)
print(median)

# apply groupby function
df = pd.read_csv("finance_liquor_sales.csv")
cn = df.groupby('category_name')
print(cn.first())

cnc = df.groupby(['category_name', 'city'])
print(cnc.first())

print(cn.aggregate(np.sum))

# Apply different aggregation methods for different columns passing a dictionary

cn2 = df.groupby(['category_name','city'])
print(cn2.agg({'bottles_sold': 'sum', 'sale_dollars': 'mean'}))

#------------------- Pandas - Concatenate/Merge and Join--------------------------------------------------------------------------

d1 = {
    'Name':['Mary','John','Alice','Bob'],
    'Age': [27,24,22,32],
    'Position':['Data Analyst', 'Trainee', 'QATester', 'IT']
}

d2 = {
    'Name': ['Steve', 'Tom', 'Jenny', 'Nick'],
    'Age': [37,25,24,52],
    'Position': ['IT', 'Data Analyst', 'Consultant', 'IT']
}

df1 = pd.DataFrame(d1, index = [0,1,2,3])
df2= pd.DataFrame(d2, index = [4,5,6,7])
final_df = pd.concat([df1,df2])
print(final_df)

d3 = {
    'key': ['a', 'b', 'c', 'd'],
    'Name': ['Mary', 'John', 'Alice', 'Bob']
}

d4 = {
    'key': ['a', 'b', 'c', 'd'],
    'Age': [27,24,22,32]
}

df3 = pd.DataFrame(d3)
df4 = pd.DataFrame(d4)
final_df2 = pd.merge(df3, df4, on='key')

d5 = {'Name': ['Mary', 'John', 'Alice', 'Bob'],
         'Age': [27, 24, 42, 32]}
d6 = {'Position': ['Data Analyst', 'Trainee', 'QA Tester', 'IT'],
          'Years_of_experience':[5, 1, 10, 3] }

df5 = pd.DataFrame(d5, index = [0,1,2,3])
df6 =  pd.DataFrame(d6, index = [0,2,3,4])
final_df3 = df5.join(df6,how = 'outer')
print(final_df3)

final_df3 = df5.join(df6,how = 'inner')
print(final_df3)


#------------------- Pandas - Hands On Tasks---------------------------------------------------------------

#Task 1

import pandas as pd
L = [5,10,15,25.,25]
S = pd.Series(L)
print(S)

# Task 2
d = {
    'col1': [1, 2, 3, 4, 7, 11],
    'col2': [4, 5, 6, 9, 5, 0],
    'col3': [7, 5, 8, 12, 1,11]
}

df = pd.DataFrame(d)
s1= df.iloc[:,0]
print("1st column as a Series:")
print(s1)
print(type(s1))

# Task 3

file = pd.read_csv("data.csv")
print(file.head(20))
print(file.shape)

# Task 4

for i, j in file.iterrows():
    print(i,j)


#------------------- Pandas - Challenge 1---------------------------------------------------------------

data = pd.read_csv("1.supermarket.csv")
print(data.info())
print(data.shape)
print(data.head())
print(data.tail())


print(data.columns)

x= data.groupby('item_name')
x= x.sum()
print(x.sum)


#------------------- Matplotlib - Introduction --------------------------------------------------------------
import matplotlib.pyplot as plt

plt.plot([0, 10, 12], [0, 300, 200], )


plt.plot([0, 10, 12], [0, 300, 200], marker = 'o' )
plt.show()

plt.plot([0, 10, 12], [0, 300, 200], marker = 'o', ls= 'dotted' )
plt.title("Graph")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.grid()
plt.show()

#Plot more than one graphs in the same figure

#Side by side in the same row, but different columns (1,2)
plt.subplot(1, 2, 1)
plt.plot([0, 2, 4, 6, 8, 10], [3, 8, 1, 10, 5, 12])

plt.subplot(1,2 , 2)
plt.plot([0, 10], [0, 300])

plt.show()

# One above the other in the same column but different rows (2,1)
plt.subplot(2, 1, 1)
plt.plot([0, 2, 4, 6, 8, 10], [3, 8, 1, 10, 5, 12])

plt.subplot(2,1,2)
plt.plot([0, 10], [0, 300])

plt.show()

#Scatter

x = np.array([99, 86, 87, 88, 111, 86,
              103, 87, 94, 78, 77, 85, 86])

y = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])

plt.scatter(x, y)

plt.show()

#Multiple scatterplots

x = np.array([99, 86, 87, 88, 111, 86,
              103, 87, 94, 78, 77, 85, 86])
y = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
plt.scatter(x, y)

x = np.array([100, 105, 84, 105, 90, 99,
              90, 95, 94, 100, 79, 112, 91, 80, 85])
y = np.array([2, 2, 8, 1, 15, 8, 12, 9,
              7, 3, 11, 4, 7, 14, 12])
plt.scatter(x, y)

plt.show()


#Bar chart

x = np.array(["A", "B", "C", "D"])

y = np.array([4, 5, 1, 10])

plt.bar(x, y)

plt.show()

#Pie chart

mylabels = np.array(["Potatoes",
                     "Bacon", "Tomatoes", "Sausages"])

x = np.array([25, 35, 15, 25])

plt.pie(x, labels=mylabels)
plt.legend()

plt.show()

#------------------- Matplotlib - Practical Example --------------------------------------------------------------

import matplotlib.pyplot as plt

age = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

cardiac_cases = [5, 15, 20, 40, 55, 55, 70, 80, 90, 95]

survival_chances = [99, 99, 90, 90, 80, 75, 60, 50, 30, 25]

plt.xlabel("Age")
plt.ylabel("Percentage")

plt.plot(age, cardiac_cases, marker = 'o', markerfacecolor = 'red', color = 'black', linewidth = 2, label = 'Cardiac Cases',
         markersize = 12)

plt.plot(age, survival_chances, color = 'yellow', linewidth = 2, label = 'Survival Cases', marker = 'o', markerfacecolor = 'green',
         markersize = 12)

plt.legend(loc='lower right', ncol=1)
plt.show()

#------------------- Numpy & Matplotlib - Practical Example --------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

products = np.array([
    ["Apple", "Orange"],
    ["Beef", "Chicken"],
    ["Candy", "Chocolate"],
    ["Fish", "Bread"],
    ["Eggs", "Bacon"]])

#create a randomized array to select 5 products

random = np.random.randint(2, size=5)

#create a list of random products that will be used

choices = []

counter = 0
for product in products:
    choices.append(product[random[counter]])
    counter +=1
print(choices)

# create a random array of percentages

percentages = []

for i in range(4):
    percentages.append(np.random.randint(10))
percentages.append(100 - sum(percentages))

print(percentages)

#create the pie chart

plt.pie(percentages, labels=choices)
plt.legend(loc='lower right', ncol=1)

plt.show()

#------------------- Matplotlib - Practical Example --------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("1.supermarket.csv")
data.info()

q = data.groupby("item_name").quantity.sum()

print(q)
plt.bar(q.index, q, color = ['orange', 'purple', 'yellow', 'red', 'green', 'blue', 'cyan'])
plt.xlabel("Items")
plt.xticks(rotation = 6)
plt.ylabel("Number of Times Ordered")
plt.title("Most Ordered Supermarket\'s Items")
plt.show()

#------------------- Web scraping - Scrap Wikipedia --------------------------------------------------------------
import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, "html.parser")
print(s.prettify())
print(s.title)
print(s.title.string)

#------------------- Web scraping- Tags --------------------------------------------------------------

import requests
from bs4 import BeautifulSoup
url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, "html.parser")
tables = s.find_all('table')
print(tables)



