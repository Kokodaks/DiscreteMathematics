from inverse_matrix_1 import inverse_via_determinant
from inverse_matrix_2_gauss import gauss_jordan_inverse

EPS = 1e-9

def read_matrix():
    while True:
        try:
            n = int(input("정사각행렬의 크기 n 을 입력하세요: ").strip())
            if n <= 0:
                print("n은 양의 정수여야 합니다.")
                continue
            break
        except ValueError:
            print("올바른 정수를 입력하세요.")

    print(f"{n}×{n} 행렬을 행 단위로 입력하세요. (예: '1 2 3')")
    A = []
    for r in range(n):
        while True:
            row_str = input(f"{r+1}번째 행: ").strip()
            try:
                row = [float(x) for x in row_str.split()]
                if len(row) != n:
                    print(f"정확히 {n}개의 값을 입력하세요.")
                    continue
                A.append(row)
                break
            except ValueError:
                print("숫자만 공백으로 구분해 입력하세요.")
    return A

def print_matrix(M, fmt="{:10.6f}"):
    if M is None:
        print("None")
        return
    for row in M:
        print(" ".join(fmt.format(x) for x in row))

def almost_equal(A, B, eps=EPS):
    if A is None or B is None:
        return A is None and B is None
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False
    n, m = len(A), len(A[0])
    for i in range(n):
        for j in range(m):
            if abs(A[i][j] - B[i][j]) > eps:
                return False
    return True

def main():
    print("=== 역행렬 비교: (1) 행렬식/수반행렬 vs (2) 가우스-조던 ===")
    A = read_matrix()

    try:
        inv1 = inverse_via_determinant(A)
        ok1 = inv1 is not None
    except Exception as e:
        inv1 = None
        ok1 = False
        print(f"\n[Det+Adj] 오류: {e}")

    try:
        inv2 = gauss_jordan_inverse(A)
        ok2 = inv2 is not None
    except Exception as e:
        inv2 = None
        ok2 = False
        print(f"\n[Gauss-Jordan] 오류: {e}")

    print("\n입력 행렬 A:")
    print_matrix(A)

    print("\n[행렬식 방식] A^{-1}:")
    print_matrix(inv1)

    print("\n[Gauss-Jordan 소거법 방식] A^{-1}:")
    print_matrix(inv2)

    if ok1 and ok2:
        same = almost_equal(inv1, inv2, eps=EPS)
        print(f"\n=> 결과 비교 (eps={EPS}): {'동일' if same else '다름'}")
    else:
        print("\n=> 두 방식 중 하나 이상이 역행렬 계산에 실패했습니다.")

if __name__ == "__main__":
    main()
