def findMinDiff(arr, m):
    # Sort the array
    arr.sort()
    
    # Initialize the minimum difference to a large value
    min_diff = float('inf')
    
    # Traverse the array, considering each subarray of size m
    for i in range(len(arr) - m + 1):
        diff = arr[i + m - 1] - arr[i]  # Difference between the max and min in the window
        
        # Update the minimum difference if a smaller difference is found
        if diff < min_diff:
            min_diff = diff
    
    return min_diff
# Test Case 1
arr1 = [7, 3, 2, 4, 9, 12, 56]
m1 = 3
print("Minimum Difference is", findMinDiff(arr1, m1))  # Output: 2

# Test Case 2
arr2 = [3, 4, 1, 9, 56, 7, 9, 12]
m2 = 5
print("Minimum Difference is", findMinDiff(arr2, m2))  # Output: 6
