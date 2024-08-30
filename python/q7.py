from bisect import bisect_left

def successfulPairs(spells, potions, success):
    # Sort the potions array for binary search
    potions.sort()
    n = len(potions)
    pairs = []
    
    for spell in spells:
        # Find the minimum potion strength required
        min_potion_strength = (success + spell - 1) // spell
        
        # Use binary search to find the first potion that is >= min_potion_strength
        index = bisect_left(potions, min_potion_strength)
        
        # Calculate the number of successful pairs
        successful_pair_count = n - index
        pairs.append(successful_pair_count)
    
    return pairs

# Example usage:
spells1 = [5,1,3]
potions1 = [1,2,3,4,5]
success1 = 7
print(successfulPairs(spells1, potions1, success1))  # Output: [4, 0, 3]

spells2 = [3,1,2]
potions2 = [8,5,8]
success2 = 16
print(successfulPairs(spells2, potions2, success2))  # Output: [2, 0, 2]
