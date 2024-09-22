import warp as wp
import numpy as np

@wp.kernel
def simple_kernel(a: wp.array(dtype=wp.vec3),
                  b: wp.array(dtype=wp.vec3),
                  c: wp.array(dtype=float)):

    # get thread index
    tid = wp.tid()

    # load two vec3s
    x = a[tid]
    y = b[tid]

    # compute the dot product between vectors
    r = wp.dot(x, y)
    # sigmoid function
    r = 1.0 / (1.0 + wp.exp(-r))

    # write result back to memory
    c[tid] = r

n = 1024

# allocate an uninitialized array of vec3s
a = np.random.rand(n, 3).astype(np.float32)
b = np.random.rand(n, 3).astype(np.float32)
c = np.zeros(n, dtype=np.float32)

# allocate and initialize an array from a NumPy array
# will be automatically transferred to the specified device
a = wp.from_numpy(a, dtype=wp.vec3, device="cuda")
b = wp.from_numpy(b, dtype=wp.vec3, device="cuda")
c = wp.from_numpy(c, dtype=wp.float32, device="cuda")


wp.launch(simple_kernel, dim=1024, inputs=[a, b, c])
