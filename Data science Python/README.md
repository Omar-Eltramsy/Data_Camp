# Chapter 1: Matplot
first thing we need to import library
```
import matplotlib.pyplot as plt
```
there is typt of plots like line or scatter.<br>
**line plot** we use it when we need to Tracking changes over time to show line plot
```
plt.plot(x-axis,y-axis)
```
**scatter plot** we use it to identify correlations
```
plt.scatter(x-axis,y-axis)
```
**Hystogram plot** we use it to get idea about distribution
```
plt.hist()
```
to show the plot
```
plt.show()
```
to label the plot
```
plt.xlabel()
plt.ylabel()
plt.title()
```
to specify the value in axis 
```
plt.yticks([ , , , ])
```
to change the scale of axis
```
plt.xscale('type of scale you want')
```
to add value in the plot 
```
plt.text(x-val,y-val,s:str)
```

# Chapter 2: dict and panda
**Dict:** list=[ , , , ] it is not acceptable to change the value in it after make it 
dict={'key':value,'key':value,---} you can change the value of any key and you can insert new key but must be each key is unique and value 
```
dict['new key']=value
```
if you want to delete key 
```
del(dict['key'])
``` 
to get the value of the key 
```
dict['key']>> value
```
you can make dict in dict
```
dict={'keyout':{'keyin':value,'keyin':value,---},
      'keyout2':{'keyin1':value,'keyin':value,---}
      'keyout3':{'keyin2':value,'keyin':value,---}
}
```
if you want to get the value of the iner dict
```
dict['keyout']['keyin']
```
**Panda**
to import panda
```
import panda as pd
```
to create dataframe from dict
```
object=pd.DataFrame(dict_name)
```
to specify index of data frame
```
dataframe_name.index=[]
```
you can read from csv file
```
dataframe_name=pd.read_csv('name of file or the path of file',index_col=0)
```
to search about row
```
dataframe_name.loc['name of row'] >>to show data in row
```
**or**
```
dataframe_name.loc[['name of row']] >>to show data in colum
```
to show the row with index
```
dataframe_name.iloc[[index]]
```
to filter the data frame 
```
dataframe[condition]
```
**like**
```
dataframe[np.logical_and(dataframe['']>10,dataframe['']<20)]
```
we use np for filter when we need to filter more than one item 
to make for loop in dict we use item
```
world={"key":value,"key1":value1}
for key,value in world.item():
    print(key + str(value))
```
and if you want to print numpy array 
```
np.nditer(array)
```
there is a way to loop in numpay array
```
for lab,row in brics.iterrows():
```

___
# Data Manipulation with pandas
## Chapter 1
to know the dimansion of data frame
```
dataframe.shape() >> (x,y) 
```
to know the name of column
```
dataframe.columns()
```
When you get a new DataFrame to work with, the first thing you need to do is explore it and see what it contains. There are several useful methods and attributes for this.
<br>
```.head()``` returns the first few rows (the “head” of the DataFrame).<br>
```.info()``` shows information on each of the columns, such as the data type and number of missing values.<br>
```.shape``` returns the number of rows and columns of the DataFrame.<br>
```.describe()``` calculates a few summary statistics for each column.<br>

To better understand DataFrame objects, it's useful to know that they consist of three components, stored as attributes:<br>
```.values```: A two-dimensional NumPy array of values.<br>
```.columns```: An index of columns: the column names.<br>
```.index```: An index for the rows: either row numbers or row names.<br>

Finding interesting bits of data in a DataFrame is often easier if you change the order of the rows. You can sort the rows by passing a column name to ```.sort_values()```.<br>
In cases where rows have the same value (this is common if you sort on a categorical variable), you may wish to break the ties by sorting on another column. You can sort on multiple columns in this way by passing a list of column names.<br>
one column	```df.sort_values("breed")```<br>
multiple columns	```df.sort_values(["breed", "weight_kg"])```
## Chapter 2
Summary statistics are exactly what they sound like - they summarize many numbers in one statistic. For example ```.mean```, ```.median```, ```.min```, ```max```, and ```.std(standard deviation)```are summary statistics. Calculating summary statistics allows you to get a better sense of your data, even if there's a lot of it.

While pandas and NumPy have tons of functions, sometimes, you may need a different function to summarize your data.

The ```.agg()``` method allows you to apply your own custom functions to a DataFrame, as well as apply functions to more than one column of a DataFrame at once, making your aggregations super-efficient. For example,
```
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)    
print(sales['temperature_c'].agg(iqr))
```
```
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)
print(sales[["temperature_c", "fuel_price_usd_per_l", "unemployment"]].agg([iqr,np.median]))
```
to sure the data dont has a duplicate data 
```
df.drop_duplicates[subset='column']
```
to count the apper of something 
```
df['column'].value_counts()
# to sort it 
df['column'].value_counts(sort=True)
# to get the percentage
df['column'].value_counts(normlize=True)
```
to group data
```
df.groupby()
```
In pandas, pivot tables are essentially another way of performing grouped calculations. That is, the ```.pivot_table()``` method is an alternative to ```.groupby()```<br>
```
import numpy as np
mean_sales_by_type_holiday = sales.pivot_table(values='weekly_sales',index='type',columns='is_holiday',aggfunc=np.mean)
print(mean_sales_by_type_holiday)
```

The ```.pivot_table()``` method has several useful arguments, including fill_value and margins.
```fill_value ```replaces missing values with a real value (known as imputation). What to replace missing values with is a topic big enough to have its own course (Dealing with Missing Data in Python), but the simplest thing to do is to substitute a dummy value.
``` margins``` is a shortcut for when you pivoted by two variables, but also wanted to pivot by each of those variables separately: it gives the row and column totals of the pivot table contents.
In this exercise, you'll practice using these arguments to up your pivot table skills, which will help you crunch numbers more efficiently!
-to get the overview of data sum of columns and rows-

## Chapter 3

