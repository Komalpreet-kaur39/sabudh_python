def extract_continuous_elements(lst, num_elements):
    result = []
    n = len(lst)

    # Loop through the list to find sublists of length `num_elements`
    for i in range(n - num_elements + 1):
        # Extract the sublist
        sublist = lst[i:i + num_elements]

        # Check if the sublist has consecutive numbers
        if all(sublist[j] + 1 == sublist[j + 1] for j in range(len(sublist) - 1)):
            result.append(sublist)

    return result

# Original list
original_list = [1, 1, 3, 4, 4, 5, 6, 7]

# Number of elements to extract
num_elements = 2

# Extract continuous elements
extracted_elements = extract_continuous_elements(original_list, num_elements)

# Print the result
print(f"Original list: {original_list}")
print(f"Extracted {num_elements} number of elements which follow each other continuously: {extracted_elements}")
