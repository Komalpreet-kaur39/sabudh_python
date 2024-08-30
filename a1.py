def shuffle_array(arr):
    n = len(arr) // 2
    temp = [0] * len(arr)
    
    for i in range(n):
        temp[2 * i] = arr[i]           
        temp[2 * i + 1] = arr[n + i]   

    for i in range(len(arr)):
        arr[i] = temp[i]

    return arr


arr1 = [1, 2, 9, 15]
arr2 = [1, 2, 3, 4, 5, 6]
arr3=  [2,4,6,8]
arr4=  [2,3,4,5,6,7,8,9]


print(shuffle_array(arr1))  # Output: [1, 9, 2, 15]
print(shuffle_array(arr2))  # Output:[1, 4, 2, 5, 3, 6]
print(shuffle_array(arr3))  #Output: [2, 6, 4, 8]
print(shuffle_array(arr4))  #Output: [2, 6, 3, 7, 4, 8, 5, 9]
