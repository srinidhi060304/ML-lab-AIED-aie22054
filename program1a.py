#Write a program to count the numberof vowels and consonants present in an input string.
def count_vowels_and_consonants(input_string):
    
    #Count the number of vowels and consonants in the given input string.
    
    #Parameters:
    #input_string (str): The input string for which the count is to be calculated.
    
    #Returns:
    #tuple: A tuple containing two values - the count of vowels and the count of consonants.
    # Initialize counters for vowels and consonants
    vowel_count = 0
    consonant_count = 0
    
    # Define a set of vowels for efficient checking
    vowels = set("aeiouAEIOU")
    # Iterate through each character in the input string
    for char in input_string:
        # Check if the character is an alphabet
        if char.isalpha():
            # Increment the vowel count if the character is a vowel
            if char in vowels:
                vowel_count += 1
            # Increment the consonant count if the character is a consonant
            else:
                consonant_count += 1
    
    # Return the counts as a tuple
    return vowel_count, consonant_count
# Main program
if __name__ == "__main__":
    # Get user input
    user_input = input("Enter a string: ")
    
    # Call the function to count vowels and consonants
    vowels, consonants = count_vowels_and_consonants(user_input)
    
    # Print the results
    print(f"Number of vowels: {vowels}")
    print(f"Number of consonants: {consonants}")
