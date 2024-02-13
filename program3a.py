import numpy as np
import pandas as pd
data=pd.read_excel("C:\\Users\\admin\\Desktop\\sem4\\Lab Session1 Data.xlsx")
#below are the answers to Question 1
Acol=data.iloc[:,1:4]
Arow=Acol.iloc[0:10,:]
A=Arow.to_numpy()
print(A)
Caccess=data.iloc[:,4]
C=Caccess.to_numpy()
print(C)
n_col,n_rows=A.shape
print("a)dimensionality of the vector space of A is:",n_col)
print("b)number of vectors in the vector space of A is:",n_rows)
# Find the rank of the matrix
rank = np.linalg.matrix_rank(A)
# Print the rank
print("c)Rank of the matrix:", rank)
Ainv=np.linalg.pinv(A)
print("d)A's psudo inverse:",Ainv)
#below are the answers to question 2
Ainv_dot_C=Ainv@C
print("Ainv dot product with C is:",Ainv_dot_C)
classification=np.zeros((10,1))
for i in C:
    j=C[1].index[i]
    if i>200:
        classification[j][1]="RICH"
    else:
        classification[j][i]="POOR"
