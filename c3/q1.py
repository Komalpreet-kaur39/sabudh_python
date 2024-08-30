def threeSum(nums):
    nums.sort()  # Step 1: Sort the array
    res = []
    
    for i in range(len(nums)):
        # Skip the same element to avoid duplicate triplets
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        # Two-pointer approach
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            
            if total == 0:
                res.append([nums[i], nums[left], nums[right]])
                
                # Move pointers to skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                # Move both pointers
                left += 1
                right -= 1
            elif total < 0:
                left += 1  # Increase the sum by moving the left pointer
            else:
                right -= 1  # Decrease the sum by moving the right pointer
    
    return res
# Test Case 1
nums1 = [-1,0,1,2,-1,-4]
print(threeSum(nums1))  # Output: [[-1,-1,2],[-1,0,1]]

# Test Case 2
nums2 = [0,1,1]
print(threeSum(nums2))  # Output: []

# Test Case 3
nums3 = [0,0,0]
print(threeSum(nums3))  # Output: [[0,0,0]]
