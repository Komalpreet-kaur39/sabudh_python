def shuffle_array(arr):
    n = len(arr) // 2 
    temp = [0] * len(arr)
    for i in range(n):
        temp[2 * i] = arr[i]  
        temp[2 * i + 1] = arr[n + i]  
    for i in range(len(arr)):
        arr[i] = temp[i]

# Test cases
arr1 = [1, 2, 9, 15]
arr2 = [1, 2, 3, 4, 5, 6]

shuffle_array(arr1)
print(arr1)  

shuffle_array(arr2)
print(arr2)  
