def findMedianSortedArrays(a, b):
    # Ensure a is the smaller array
    if len(a) > len(b):
        a, b = b, a

    n, m = len(a), len(b)
    imin, imax, half_len = 0, n, (n + m + 1) // 2

    while imin <= imax:
        i = (imin + imax) // 2
        j = half_len - i

        if i < n and b[j-1] > a[i]:
            imin = i + 1  # i is too small, must increase it
        elif i > 0 and a[i-1] > b[j]:
            imax = i - 1  # i is too large, must decrease it
        else:
            # i is perfect
            if i == 0: max_of_left = b[j-1]
            elif j == 0: max_of_left = a[i-1]
            else: max_of_left = max(a[i-1], b[j-1])

            if (n + m) % 2 == 1:
                return max_of_left  # Odd total length, max of left part is the median

            if i == n: min_of_right = b[j]
            elif j == m: min_of_right = a[i]
            else: min_of_right = min(a[i], b[j])

            return (max_of_left + min_of_right) / 2.0  # Even total length, median is the average

# Example usage:
a1 = [-5, 3, 6, 12, 15]
b1 = [-12, -10, -6, -3, 4, 10]
print("The median is", findMedianSortedArrays(a1, b1))  # Output: 3

a2 = [2, 3, 5, 8]
b2 = [10, 12, 14, 16, 18, 20]
print("The median is", findMedianSortedArrays(a2, b2))  # Output: 11
