def maxProductTriplet(arr):
    # Sort the array
    arr.sort()
    
    # The triplet can be either:
    # 1. The product of the three largest elements
    max_product1 = arr[-1] * arr[-2] * arr[-3]
    
    # 2. The product of the two smallest elements and the largest element
    max_product2 = arr[0] * arr[1] * arr[-1]
    
    # Determine which product is greater
    if max_product1 > max_product2:
        return (arr[-3], arr[-2], arr[-1])
    else:
        return (arr[0], arr[1], arr[-1])

# Example usage:
arr1 = [-4, 1, -8, 9, 6]
result1 = maxProductTriplet(arr1)
print(f"The triplet having the maximum product is {result1}")

arr2 = [1, 7, 2, -2, 5]
result2 = maxProductTriplet(arr2)
print(f"The triplet having the maximum product is {result2}")
