#행렬식을 이용한 역행렬

def minor_matrix(M, i, j):
    return [row[:j] + row[j+1:] for idx, row in enumerate(M) if idx != i]

def determinant(M):
    n = len(M)
    if n == 0:
        return 1.0
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0]*M[1][1] - M[0][1]*M[1][0]

    det = 0.0

    for j in range(n):
        cofactor_sign = -1 if (0 + j) % 2 else 1
        det += cofactor_sign * M[0][j] * determinant(minor_matrix(M, 0, j))
    return det

def cofactor_matrix(M):
    n = len(M)
    C = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sign = -1 if (i + j) % 2 else 1
            C[i][j] = sign * determinant(minor_matrix(M, i, j))
    return C

def transpose(M):
    return [list(row) for row in zip(*M)]

def adjugate(M):
    return transpose(cofactor_matrix(M))

def inverse_via_determinant(M, eps=1e-12):
    if len(M) == 0 or len(M) != len(M[0]):
        raise ValueError("정방행렬만 허용됩니다.")

    detA = determinant(M)
    if abs(detA) < eps:
        return None

    adjA = adjugate(M)
    n = len(M)
    invA = [[adjA[i][j] / detA for j in range(n)] for i in range(n)]
    return invA