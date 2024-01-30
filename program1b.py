#Write a program that accepts two matrices A and Bas input and returns their product AB.Check if A & B aremultipliable; if not, return error message.
def input_matrix(rows, cols):
    """
    Function to input a matrix from the user.
    
    Parameters:
    - rows: Number of rows in the matrix.
    - cols: Number of columns in the matrix.
    
    Returns:
    - A 2D list representing the matrix.
    """
    matrix = []
    for i in range(rows):
        row = [float(input(f"Enter element at position ({i+1},{j+1}): ")) for j in range(cols)]
        matrix.append(row)
    return matrix

def matrix_multiply(A, B):
    """
    Function to multiply two matrices A and B.

    Parameters:
    - A: First matrix (m x p)
    - B: Second matrix (p x n)

    Returns:
    - Resulting matrix (m x n) if A and B are multipliable.
    - None if A and B are not multipliable.
    """
    # Check if matrices are multipliable
    if len(A[0]) != len(B):
        return None
    
    # Initialize the result matrix with zeros
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    
    # Matrix multiplication
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

if __name__ == "__main__":
    # Input matrix A
    rowsA = int(input("Enter the number of rows for matrix A: "))
    colsA = int(input("Enter the number of columns for matrix A: "))
    matrix_A = input_matrix(rowsA, colsA)

    # Input matrix B
    rowsB = int(input("Enter the number of rows for matrix B: "))
    colsB = int(input("Enter the number of columns for matrix B: "))
    matrix_B = input_matrix(rowsB, colsB)

    # Multiply matrices A and B
    result_matrix = matrix_multiply(matrix_A, matrix_B)

    # Display the result or an error message
    if result_matrix is not None:
        print("\nMatrix A:")
        for row in matrix_A:
            print(row)

        print("\nMatrix B:")
        for row in matrix_B:
            print(row)

        print("\nResultant Matrix AB:")
        for row in result_matrix:
            print(row)
    else:
        print("Error: Matrices A and B are not multipliable.")
