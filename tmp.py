N = 4

for i in range(1, N):
    for j in range(i):
        print(f"({i-j:2d}, {j:2d}), ({i-j-1:2d}, {j:2d}),", end=" ")
    print(f"({i-j-1:2d}, {j+1:2d})")
    # print(j)

    for j in range(i):
        print(f"({i-j:2d}, {j+1:2d}),", end=" ")

    print()
    # break

for i in range(N, 2 * N - 2):
    for j in range(i - N + 1, N - 1):
        print(f"({i-j:2d}, {j:2d}), ({i-j-1:2d}, {j:2d})", end=" ")
    print(f"({i-j-1:2d}, {j+1:2d})")

    for j in range(i - N + 1, N - 1):
        # print(j, end=" ")
        print(f"({i-j:2d}, {j+1:2d}),", end=" ")
    print()
