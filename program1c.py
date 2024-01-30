#Write a program to find the number of common elements between two lists. The lists contain integers.
def get_user_input():
    #Function to get input lists from the user.

    #Returns:
    #- list: First input list of integers.
    #- list: Second input list of integers.
    l1 = list(map(int, input("Enter the elements of the first list separated by space: ").split()))
    l2 = list(map(int, input("Enter the elements of the second list separated by space: ").split()))
    return l1, l2


def find_com_ele(l1, l2):
    #Function to find the number of common elements between two lists.

    #Arguments:
    #- l1 (list): First list of integers.
    #- l2 (list): Second list of integers.

    #Returns:
    #- int: Number of common elements.
    com_ele = set(l1) & set(l2)
    return len(com_ele)


def main():
    # Step 1: Get input from the user
    l1, l2 = get_user_input()

    # Step 2: Find the number of common elements
    com_ele_count = find_com_ele(l1, l2)

    # Step 3: Display the result
    print(f"Number of common elements between the two lists: {com_ele_count}")


# Run the main program
if __name__ == "__main__":
    main()
