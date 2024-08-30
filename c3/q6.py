def romanToInt(s):
    # Map of Roman symbols to their values
    roman_to_int = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000
    }
    
    # Initialize the total sum
    total = 0
    
    # Iterate over the string, except the last character
    for i in range(len(s) - 1):
        # If the current character is less than the next, subtract it
        if roman_to_int[s[i]] < roman_to_int[s[i + 1]]:
            total -= roman_to_int[s[i]]
        # Otherwise, add it
        else:
            total += roman_to_int[s[i]]
    
    # Add the value of the last character
    total += roman_to_int[s[-1]]
    
    return total
# Test Case 1
s1 = "III"
print(romanToInt(s1))  # Output: 3

# Test Case 2
s2 = "LVIII"
print(romanToInt(s2))  # Output: 58

# Test Case 3
s3 = "MCMXCIV"
print(romanToInt(s3))  # Output: 1994
