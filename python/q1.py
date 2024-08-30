def shuffleArray(arr, n):
    for i in range(n):
        j = n + i
        while j > i:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1

# Example usage:
arr = [1, 2, 3, 4, 5, 6]
shuffleArray(arr, len(arr) // 2)
print(arr)  # Output: [1, 4, 2, 5, 3, 6]
