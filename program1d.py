#Write a program that accepts a matrix as input and returns its transpose.
def get_matrix():
    #Function to accept a matrix as input from the user.
    #Returns:
    #- matrix: The matrix entered by the user.
    
    # Get the number of rows and columns for the matrix
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))

    # Initialize an empty matrix
    matrix = []

    # Input elements for each row
    for i in range(rows):
        row = []
        for j in range(cols):
            element = float(input(f"Enter element at position ({i+1}, {j+1}): "))
            row.append(element)
        matrix.append(row)

    return matrix

def transpose_matrix(matrix):
    
    #Function to calculate and return the transpose of a matrix.
    #Args:
    #- matrix: The input matrix.
    #Returns:
    #- transposed_matrix: The transposed matrix.
    # Use zip to transpose the matrix
    transposed_matrix = [list(row) for row in zip(*matrix)]

    return transposed_matrix

def main():
    #Main program to execute the matrix transpose functionality.
    
    # Step 1: Get the matrix from the user
    input_matrix = get_matrix()

    # Step 2: Calculate the transpose of the matrix
    result_matrix = transpose_matrix(input_matrix)

    # Step 3: Display the result
    print("\nInput Matrix:")
    for row in input_matrix:
        print(row)

    print("\nTransposed Matrix:")
    for row in result_matrix:
        print(row)

# Execute the main program
if __name__ == "__main__":
    main()
