def countPairsWithSum(arr, K):
    # Dictionary to store frequencies of elements
    freq = {}
    count = 0
    
    # Traverse the array
    for num in arr:
        # Check if the complement (K - num) exists in the hash map
        complement = K - num
        if complement in freq:
            count += freq[complement]
        
        # Add the current number to the hash map
        if num in freq:
            freq[num] += 1
        else:
            freq[num] = 1
    
    return count
# Test Case 1
arr1 = [1, 5, 7, -1]
K1 = 6
print(countPairsWithSum(arr1, K1))  # Output: 2

# Test Case 2
arr2 = [1, 5, 7, -1, 5]
K2 = 6
print(countPairsWithSum(arr2, K2))  # Output: 3
