#가우스 조던 소거법을 이용한 역행렬 계산하는 함수

def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def gauss_jordan_inverse(A, eps=1e-12):
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("정방행렬만 허용됩니다.")

    # 확장 행렬 [A | I] 생성 (deep copy)
    aug = [list(map(float, A[i])) + identity(n)[i] for i in range(n)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        pivot_val = aug[pivot_row][col]

        if abs(pivot_val) < eps:
            raise ValueError("역행렬이 존재하지 않습니다(특이 행렬).")

        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        for j in range(2*n):
            aug[col][j] /= pivot

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor != 0.0:
                for j in range(2*n):
                    aug[r][j] -= factor * aug[col][j]

    inv = [row[n:] for row in aug]
    return inv
