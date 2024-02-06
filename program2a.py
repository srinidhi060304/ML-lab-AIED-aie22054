import math

def euclidean_distance(vector_1, vector_2):
    # Calculate the Euclidean distance between two vectors.
    # Ensure both vectors have the same dimensions
    if len(vector_1) != len(vector_2):
        raise ValueError("Vectors must have the same dimensions")
    
    # Calculate the sum of squared differences for each dimension
    sum_squared_difference = sum([(x - y) ** 2 for x, y in zip(vector_1, vector_2)])
    
    # Return the square root of the sum of squared differences
    return math.sqrt(sum_squared_difference)

def manhattan_distance(vector_1, vector_2):
    # Calculate the Manhattan distance between two vectors.
    # Ensure both vectors have the same dimensions
    if len(vector_1) != len(vector_2):
        raise ValueError("Vectors must have the same dimensions")
    # Calculate the sum of absolute differences for each dimension
    sum_absolute_difference = sum([abs(x - y) for x, y in zip(vector_1, vector_2)])
    
    # Return the sum of absolute differences
    return sum_absolute_difference

vector_1 = [1, 2, 3]
vector_2 = [4, 5, 6]

euclidean_dist = euclidean_distance(vector_1, vector_2)
manhattan_dist = manhattan_distance(vector_1, vector_2)

print("Euclidean Distance:", euclidean_dist)
print("Manhattan Distance:", manhattan_dist)
