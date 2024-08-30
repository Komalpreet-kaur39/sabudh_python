def rotate(matrix):
    n = len(matrix)
    
    # Step 1: Transpose the matrix
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # Step 2: Reverse each row
    for i in range(n):
        matrix[i].reverse()

matrix1 = [[1,2,3],
          [4,5,6],
          [7,8,9]]
rotate(matrix1)
print(matrix1)
matrix2 = [[5,1,9,11],
          [2,4,8,10],
          [13,3,6,7],
          [15,14,12,16]]
rotate(matrix2)
print(matrix2)
