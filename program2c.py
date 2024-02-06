def label_encoding(data):
    # Convert categorical variables to numeric using label encoding.
    # Create an empty dictionary to store label mappings
    lab_map = {}
    
    # Initialize label counter
    label_counter = 0
    
    # Initialize empty list to store encoded data
    encoded_data = []
    
    # Iterate through each categorical variable
    for category in data:
        # Check if category is not already in label mapping
        if category not in lab_map:
            # Add category to label mapping with corresponding label
            lab_map[category] = label_counter
            # Increment label counter for next category
            label_counter += 1
        
        # Append the encoded label to the encoded data list
        encoded_data.append(lab_map[category])
    
    return encoded_data, lab_map

# Define categorical data
categorical_data = ["human", "fish", "chicken", "fish", "human", "dog", "cat", "dog"]

# Apply label encoding
encoded_data, lab_map = label_encoding(categorical_data)

# Print encoded data and label mapping
print("Encoded data:", encoded_data)
print("Label mapping:", lab_map)
