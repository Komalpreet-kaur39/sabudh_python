def rearrange(arr):
    # Start from the second element (index 1)
    for i in range(1, len(arr)):
        key = arr[i]
        
        # If the key is negative, move it to its correct position in the front
        if key < 0:
            j = i - 1
            # Shift positive numbers to the right
            while j >= 0 and arr[j] >= 0:
                arr[j + 1] = arr[j]
                j -= 1
            
            # Place the negative number at its correct position
            arr[j + 1] = key

# Test Case 1
arr1 = [-12, 11, -13, -5, 6, -7, 5, -3, -6]
rearrange(arr1)
print(arr1)  # Output: [-12, -13, -5, -7, -3, -6, 11, 6, 5]

# Test Case 2
arr2 = [-12, 11, 13, -5, 6, -7, 5, -3, 8]
rearrange(arr2)
print(arr2)  # Output: [-12, -5, -7, -3, 11, 13, 6, 5, 8]
