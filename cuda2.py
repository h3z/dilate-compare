from numba import cuda
import pickle
import numpy as np
import numba


@cuda.jit
def increment_by_one(an_array):
    x, y = cuda.grid(2)
    arr = cuda.local.array((3, 4), dtype=numba.int32)
    arr[x, y] += 1
    an_array[x, y] = x + y + arr[x, y]
    an_array[x, :] = 0
    return
    # print(pos)
    # if pos < an_array.size:
    #     an_array[pos] += 1


if __name__ == "__main__":
    data = pickle.load(open("dump/dcpu.dump", "rb"))
    print(data.shape)
    # threadsperblock = 32
    # blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock
    # increment_by_one[blockspergrid, threadsperblock](data)
    a = np.zeros((3, 4))
    increment_by_one[(3, 4), (1, 1)](a)
    print(a)
