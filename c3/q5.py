def intToRoman(num):
    # List of Roman numerals and their corresponding integer values
    roman_symbols = [
        ("M", 1000),
        ("CM", 900),
        ("D", 500),
        ("CD", 400),
        ("C", 100),
        ("XC", 90),
        ("L", 50),
        ("XL", 40),
        ("X", 10),
        ("IX", 9),
        ("V", 5),
        ("IV", 4),
        ("I", 1)
    ]
    
    # Initialize the result string
    result = ""
    
    # Iterate through each Roman numeral
    for symbol, value in roman_symbols:
        # While the current value can be subtracted from the number
        while num >= value:
            # Append the symbol to the result
            result += symbol
            # Subtract the value from the number
            num -= value
    
    return result
# Test Case 1
num1 = 3749
print(intToRoman(num1))  # Output: "MMMDCCXLIX"

# Test Case 2
num2 = 58
print(intToRoman(num2))  # Output: "LVIII"

# Test Case 3
num3 = 1994
print(intToRoman(num3))  # Output: "MCMXCIV"
