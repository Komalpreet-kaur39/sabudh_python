# ---------------------LIST------------------------------
# Q1: Write a Python program to multiply all the items in a list.
def multiply_items(l1):
    result=1
    for i in l1:
        result*=i
    return result
list1=[2,3,4,5,6,4,6]
print("List of elements : ",list1)
result = multiply_items(list1)
print(" Multiplication of all items in the list : ",result)


# Q2: Write a Python program to get the largest number from a list.
def largest_number(l2):
    result1=0
    for i in l2:
        if i>result1:
            result1=i
    return result1
list2=[23,56,87,12,34,65]
print("\nList of elements : ",list2)
result1 = largest_number(list2)
print(" Largest number in the list : ",result1)

# Q3: Write a Python program to get the smallest number from a list.
def smallest_number(l2):
    result1=l2[0]
    for i in l2:
        if i<result1:
            result1=i
    return result1
list2=[34,56,78,12,67,87]
print("\nList of elements : ",list2)
result1 = smallest_number(list2)
print("Smallest number in the list : ",result1)

# Q4: Write a Python program to get a list, sorted in increasing order by the last element in each tuple from a given list of non-empty tuples.
def sort_elements(tuple_list):
   return sorted(tuple_list, key=lambda x: x[-1])
tuple1=[(2,3),(4,2),(2,9),(9,0)]
print("\nList of tuple",tuple1)
result=sort_elements(tuple1)
print(" sorted tuple by last index are :",result)



# Q5: Write a Python program to remove duplicates from a list.
def remove_duplicate(l1):
    result=[]
    for i in l1:
        if i not in result:
            result.append(i)
    return result
l1=[2,3,4,5,3,5,1,7,4]
print("\nList of elements:",l1)
result=remove_duplicate(l1)
print("After removing the duplicates the list is ",result)

# Q6:Write a Python program to check if a list is empty or not.
def check_empty(l2):
        if not l2:
            print("list is empty")
        else:
            print("list is not empty ",l2)
l1=[3,4,6,7,7]
print("\nList :",l1)
result=check_empty(l1)
l2=[]
print("List :",l2)
result1=check_empty(l2)

# Q7: Write a Python program to count the lowercase letters in a given list of word
def lowercase(word_list):
    count=0
    for word in word_list:
        for char in word:
            if char.islower():
                count+=1
    return count
word_list=["Python","Java","Print","Words"]
print("\nList of words :",word_list)
result=lowercase(word_list)
print("The count the lowercase letters in a given list of word : ",result)

# Q8: Write a Python program to extract specified number of elements from a given list, which follows each other continuously.
def u(a,num):
    s=""
    for i in a:
        s+=str(i)
    n=set() 
    for j in s:
        if s.count(j)==num:
          n.add(j)
    nl=list(n)
    print("Extracted number from list:",nl)

a1=[1, 1, 3, 4, 4, 5, 6, 7]
a2=[0, 1, 2, 3, 4, 4, 4, 4, 5, 7]
print("\nList of elements",a1)
u(a1,2)
print("List of elements",a2)
u(a2,4)

# Q9: Write a Python program to find the largest odd number in a given list of integers.
def largest_odd(list1):
    largest_number=0
    for number in list1:
        if number%2!=0:
            if number>largest_number:
               largest_number=number
    return largest_number
number=[2,43,5,7,8]
print("\n List :",number)
result=largest_odd(number)
print("Largest odd number in the list of integers is :",result)

# Q10:  Write a Python program to print a specified list after removing the 0th, 4th and 5th elements.
def remove_number(l2,indices):
    indices.sort(reverse=True)
    for index in indices:
        if index<len(l2):
            del l2[index]
    return l2
l1=[0,1,2,3,4,5,6]
indices=[0,4,5]
print("\nList:",l1)
result=remove_number(l1,indices)
print(" List after removing the 0th, 4th and 5th elements:",result)

# ---------------------------tuples------------------------------------
# Q1 : Write a Python program to create a tuple with different data types.
tuple1=(1,1.3,"Python",[1,3,4],(1,2,4),{"fruit":"apple"},True)
print("\n Tuple : ",tuple1)
print(type(tuple1))

# Q2: Write a Python program to create a tuple of numbers and print one item.
tuple2=(2,4,1,6,4,7,8)
print("\nTuple:",tuple2)
print("The fourth element of the tuple is : ",tuple2[3])

# Q3:  Write a Python program to add an item to a tuple.
my_tuple = (1, 2, 3)
print("\nTuple:",my_tuple)
new_item = 4
# Convert to list
my_list = list(my_tuple)
# Add item to list
my_list.append(new_item)
# Convert back to tuple
result_tuple = tuple(my_list)
print("After adding an item to a tuple",result_tuple) 

# Q4: Write a Python program to get the 4th element from the last element of a Tuple.
def get_4_element(tup):
    if(len(tup)>4):
        return tup[-4]
tuple1=(1,2,3,4,45,5)
print("\nList of tuple :",tuple1)
result=get_4_element(tuple1)
print(" 4th element from the last element of a Tuple:",result)

# Q5: Write a Python program to convert a tuple to a dictionary.
tuple1=(("fruit","banana"),("vegetable","spanish"),("colour","blue"))
dict1=dict(tuple1)
print("\ntuple to dictionary---")
print(dict1)
print(type(dict1))

# Q6: Write a Python program to replace the last value of tuples in a list.
def replace_last_tuple(tup,new_value):
    tuple2=[]
    for t in tup:
        new_tuple=t[:-1]+(new_value,)   
        tuple2.append(new_tuple)
    return tuple2
tuple1=[(12,3,4,2),(4,2,3,4),(3,4,4,2)]
print("\nTuple ",tuple1)
result=replace_last_tuple(tuple1,100)
print("After replacing the last value of tuples in a list:",result)

# ---------------------------Dictionary-------------------------------------
# Q1:Write a Python script to sort (ascending and descending) a dictionary by value
# Sample dictionary
data = {
    'apple': 10,
    'banana': 2,
    'cherry': 7,
    'date': 5
}
# Sort dictionary by value in ascending order
sorted_ascending = dict(sorted(data.items(), key=lambda item: item[1]))
# Sort dictionary by value in descending order
sorted_descending = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
# Print results
print("\nOriginal Dictionary:")
print(data)
print("Sorted by Value (Ascending):")
print(sorted_ascending)
print("Sorted by Value (Descending):")
print(sorted_descending)

# Q2:  Write a Python program to iterate over dictionaries using for loops.
dict1={
    1:'Apple',
    2:'Banana',
    3:'Cherry',
    4:'Grapes',
}
print("\nIterating over the keys:")
for key in dict1:
    print(key)
print("Iterating over the values:")
for value in dict1.values():
    print(value)
print("Iterating over both the key and values:")
for key,value in dict1.items():
    print(key,value)

# Q3: Write a Python script to merge two Python dictionaries.
# Dictionaries to merge
dict1 = {'apple': 20, 'banana': 50}
dict2 = {'cherry': 70, 'date': 40}
print("\nDict1:",dict1)
print("Dict2",dict2)
# Merging dictionaries
merged_dict = dict1 | dict2
print("Merged Dictionary (using merge operator):")
print(merged_dict)

# Q4:Write a Python program to sum all the items in a dictionary.
my_dict = {
    'a': 20, 'b': 20, 'c': 20, 'd': 40}
print("\nDict:",my_dict)
# Summing all the values in the dictionary
total_sum = sum(my_dict.values())
print("Sum of all items in the dictionary:", total_sum)

# Q5: Write a Python program to multiply all the items in a dictionary.
dict1={
    'a':1,'b':2,'c':3,'d':4
}
print("\nDict:",dict1)
product=1
for value in dict1.values():
    product*=value
print("Product of all items in the dictionary:", product)

# Q6: Write a Python program to sort a given dictionary by key.
def sort_dict_by_key(d):
    return dict(sorted(d.items()))
dict1={
    'banana':2,
    'cherry':3,
    'apple':4}
print("\nDict:",dict1)
result=sort_dict_by_key(dict1)
print("After sorting the dictionary by key :",result)

# Q7: Write a Python program to remove duplicates from the dictionary.
def remove_duplicate(d):
    new_dict={}
    for key,value in d.items():
        if value not in new_dict:
            new_dict[value]=key
    return { v: k for k, v in new_dict.items() }
dict2={
    'a': 10,
    'b': 20,
    'c': 10,
    'd': 30,
    'e': 20
}
print("\nDict:",dict2)
result=remove_duplicate(dict2)
print("After removing duplicates from the dictionary:",result)


# -----------------------Numpy----------------------------
import numpy as np

# Q1: Numpy array creation and manipulation
# 1. Create a 1D Numpy array “a” containing 10 random integers between 0 and 99.
a = np.random.randint(0, 100, size=10)
print("1D array 'a':\n", a)

# 2. Create a 2D Numpy array “b” of shape (3, 4) containing random integers between -10 and 10.
b = np.random.randint(-10, 11, size=(3, 4))
print("\n2D array 'b' (3x4):\n", b)

# 3. Reshape “b” into a 1D Numpy array “b_flat”.
b_flat = b.flatten()
print("\n1D array 'b_flat' (flattened 'b'):\n", b_flat)

# 4. Create a copy of “a” called “a_copy”, and set the first element of “a_copy” to -1.
a_copy = a.copy()
a_copy[0] = -1
print("\nOriginal 'a' array:\n", a)
print("\nModified 'a_copy' with first element set to -1:\n", a_copy)

# 5. Create a 1D Numpy array “c” containing every second element of “a”.
c = a[::2]
print("\n1D array 'c' (every second element of 'a'):\n", c)

# Q2:Numpy array indexing and slicing

# Print the third element of “a”.
third_element_a = a[2]
print("\nThird element of 'a':", third_element_a)

# Print the last element of “b”.
last_element_b = b[-1, -1]
print("\nLast element of 'b':", last_element_b)

# Print the first two rows and last two columns of “b”.
first_two_rows_last_two_cols = b[:2, -2:]
print("\nFirst two rows and last two columns of 'b':\n", first_two_rows_last_two_cols)

# Assign the second row of “b” to a variable called “b_row”.
b_row = b[1, :]
print("\nSecond row of 'b' assigned to 'b_row':\n", b_row)

# Assign the first column of “b” to a variable called “b_col”.
b_col = b[:, 0]
print("\nFirst column of 'b' assigned to 'b_col':\n", b_col)

# Q3:Numpy array operations
 
# Create a 1D Numpy array “d” containing the integers from 1 to 10.
d = np.arange(1, 11)
print("\n1D array 'd' (integers from 1 to 10):\n", d)

# 1. Add “a” and “d” element-wise to create a new Numpy array “e”.
e = a + d
print("\nArray 'e' (a + d element-wise):\n", e)

# 2. Multiply “b” by 2 to create a new Numpy array “b_double”.
b = np.random.randint(-10, 11, size=(3, 4))
b_double = b * 2
print("\nArray 'b':\n", b)
print("\nArray 'b_double' (b * 2):\n", b_double)

# 3. Calculate the dot product of “b” and “b_double” to create a new Numpy array “f”.
f = np.dot(b, b_double.T)
print("\nDot product 'f' of 'b' and 'b_double':\n", f)

# 4. Calculate the mean of “a”, “b”, and “b_double” to create a new Numpy array “g”.
mean_a = np.mean(a)
mean_b = np.mean(b)
mean_b_double = np.mean(b_double)
g = np.array([mean_a, mean_b, mean_b_double])
print("\nMean of 'a', 'b', and 'b_double' stored in array 'g':\n", g)

# Q4: Numpy array aggregation

# 1. Find the sum of every element in “a” and assign it to a variable “a_sum”.
a_sum = np.sum(a)
print("\nSum of elements in 'a' (a_sum):", a_sum)

# 2. Find the minimum element in “b” and assign it to a variable “b_min”.
b_min = np.min(b)
print("\nMinimum element in 'b' (b_min):", b_min)

# 3. Find the maximum element in “b_double” and assign it to a variable “b_double_max”.
b_double = b * 2
print("\nArray 'b_double' (b * 2):\n", b_double)
b_double_max = np.max(b_double)
print("\nMaximum element in 'b_double' (b_double_max):", b_double_max)

# -------------------------------Pandas--------------------------------
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Download the dataset (replace with your download method)
# Replace 'path/to/your/dataset.csv' with the actual path
url = "C:/Users/Maninder Singh/Downloads/archive/Sport car price.csv"
df = pd.read_csv(url)

# Display the first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Clean the dataset
# Handle missing values (consider appropriate actions based on data analysis)
df.dropna(inplace=True)  # Drop rows with missing values (replace with imputation if needed)
# Handle duplicates
df.drop_duplicates(inplace=True)  # Remove duplicate rows

# Convert non-numeric data (consider appropriate conversions)
# Example: convert price to numeric (assuming price is a string)
df['Price (in USD)'] = df['Price (in USD)'].str.replace(',', '').astype(float)
# Convert 'Horsepower' to numeric, coercing non-numeric to NaN
df['Horsepower'] = pd.to_numeric(df['Horsepower'], errors='coerce')

# Handle missing values (replace with mean)
df['Horsepower'] = df['Horsepower'].fillna(df['Horsepower'].mean())

# Convert the '0-60 MPH Time (seconds)' column to numeric, coercing errors to NaN
df['0-60 MPH Time (seconds)'] = pd.to_numeric(df['0-60 MPH Time (seconds)'], errors='coerce')

# Drop rows where the '0-60 MPH Time (seconds)' is NaN
df.dropna(subset=['0-60 MPH Time (seconds)'], inplace=True)

# Explore the dataset
print("\nSummary statistics:")
print(df.describe())

# Group by car make and calculate average price
avg_price_by_make = df.groupby('Car Make')['Price (in USD)'].mean()
print("\nAverage price by car make:")
print(avg_price_by_make)

# Group by year and calculate average horsepower
avg_hp_by_year = df.groupby('Year')['Horsepower'].mean()
print("\nAverage horsepower by year:")
print(avg_hp_by_year)

# Fit linear regression model using DataFrame
model = LinearRegression()
model.fit(df[['Horsepower']], df['Price (in USD)'])

# Generate predictions using the same DataFrame
y = model.predict(df[['Horsepower']])

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(df['Horsepower'], df['Price (in USD)'], label='Data Points')
plt.plot(df['Horsepower'], y, color='red', label='Regression Line')
plt.xlabel('Horsepower')
plt.ylabel('Price (in USD)')
plt.title('Price vs Horsepower (with linear regression)')
plt.legend()
plt.grid(True)
plt.show()

# Histogram of 0-60 MPH times with bins of 0.5 seconds
plt.figure(figsize=(8, 5))
plt.hist(df['0-60 MPH Time (seconds)'], bins=np.arange(0, df['0-60 MPH Time (seconds)'].max() + 0.5, 0.5))
plt.xlabel('0-60 mph (seconds)')
plt.ylabel('Number of cars')
plt.title('Histogram of 0-60 mph Times')
plt.grid(True)
plt.show()

# Filter for price > $500,000 and sort by horsepower (descending)
filtered_df = df[df['Price (in USD)'] > 500000].sort_values(by='Horsepower', ascending=False)
print("\nFiltered dataset (Price > $500,000, sorted by Horsepower):")
print(filtered_df)

# Export cleaned and transformed dataset to CSV
filtered_df.to_csv('cleaned_sports_car_prices.csv', index=False)

print("\nData cleaned, analyzed, and exported to 'cleaned_sports_car_prices.csv'")