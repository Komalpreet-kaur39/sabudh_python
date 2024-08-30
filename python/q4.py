def trapWater(arr):
    n = len(arr)
    if n < 3:
        return 0  # No water can be trapped if there are less than 3 bars

    left, right = 0, n - 1
    leftMax, rightMax = arr[left], arr[right]
    trappedWater = 0

    while left < right:
        if leftMax < rightMax:
            left += 1
            leftMax = max(leftMax, arr[left])
            trappedWater += max(0, leftMax - arr[left])
        else:
            right -= 1
            rightMax = max(rightMax, arr[right])
            trappedWater += max(0, rightMax - arr[right])

    return trappedWater

# Example usage:
arr1 = [0, 2, 0, 2, 0]
print("Water trapped:", trapWater(arr1))  # Output: 2

arr2 = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1, 0]
print("Water trapped:", trapWater(arr2))  # Output: 6
