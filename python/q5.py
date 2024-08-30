def nextPermutation(nums):
    n = len(nums)
    i = n - 2
    
    # Step 1: Find the first decreasing element from the end
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    
    if i >= 0:  # If such element is found
        # Step 2: Find the element just larger than nums[i]
        j = n - 1
        while nums[j] <= nums[i]:
            j -= 1
        # Swap them
        nums[i], nums[j] = nums[j], nums[i]
    
    # Step 3: Reverse the elements after the i-th index
    nums[i + 1:] = reversed(nums[i + 1:])
    
# Example usage:
nums1 = [1, 2, 3]
nextPermutation(nums1)
print(nums1)  # Output: [1, 3, 2]

nums2 = [3, 2, 1]
nextPermutation(nums2)
print(nums2)  # Output: [1, 2, 3]

nums3 = [1, 1, 5]
nextPermutation(nums3)
print(nums3)  # Output: [1, 5, 1]
