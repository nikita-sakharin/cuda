#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <thrust/extrema.h>
#include <thrust/device_vector.h>

typedef signed char schar;
typedef unsigned char uchar;
typedef short shrt;
typedef unsigned short ushrt;
typedef unsigned uint;
typedef unsigned long ulong;
typedef long long llong;
typedef unsigned long long ullong;

typedef float flt;
typedef double dbl;
typedef long double ldbl;

class Compare
{
public:
    __host__ __device__ bool operator()(const dbl a, const dbl b) const
    {
        return fabs(a) < fabs(b);
    }
};
__global__ static void row_switching(dbl * __restrict__, uint,
    uint, uint);
__global__ static void column_to_null_down(dbl * __restrict__, uint,
    uint);
__global__ static void column_to_null_up(dbl * __restrict__, uint,
    uint);
__global__ static void solve(dbl * __restrict__, uint);

__host__ static uint matrix_in(dbl ** __restrict__);
__host__ static void matrix_out(dbl * __restrict__, uint);
__host__ static void gauss_jordan_elimination(dbl * __restrict__, uint) noexcept;

int main()
{
    dbl *matrix;
    const uint n = matrix_in(&matrix);

    gauss_jordan_elimination(matrix, n);

    matrix_out(matrix, n);

    return 0;
}

__host__ static void gauss_jordan_elimination(
    dbl * __restrict__ const host_matrix, const uint n) noexcept
{
    const Compare compare;
    const dim3 block = dim3(32U, 16U), thread = dim3(32U, 16U);

    dbl *device_matrix;
    cudaMalloc(&device_matrix, sizeof(dbl) * n * (n + 1));
    cudaMemcpy(device_matrix, host_matrix, sizeof(dbl) * n * (n + 1),
        cudaMemcpyHostToDevice);
    const thrust::device_ptr<dbl> ptr = thrust::device_pointer_cast(device_matrix);

    for (uint i = 0; i < n - 1; ++i)
    {
        const uint max_idx = thrust::max_element(
            ptr + i * n + i,
            ptr + (i + 1) * n, compare) - ptr - i * n;
        if (max_idx != i)
        {
            row_switching<<<512U, 512U>>>(device_matrix, n, i, max_idx);
        }
        column_to_null_down<<<block, thread>>>(device_matrix, n, i);
    }
    for (uint i = n - 1; i > 0; --i)
    {
        column_to_null_up<<<512U, 512U>>>(device_matrix, n, i);
    }

    solve<<<512U, 512U>>>(device_matrix, n);
    cudaMemcpy(host_matrix + n * n, device_matrix + n * n, sizeof(dbl) * n,
        cudaMemcpyDeviceToHost);

    cudaFree(device_matrix);
}

__global__ static void row_switching(dbl * __restrict__ const m,
    const uint n, const uint i, const uint j)
{
    const uint idx = threadIdx.x + blockDim.x * blockIdx.x,
        offset = blockDim.x * gridDim.x;
    for (uint k = i + idx; k <= n; k += offset)
    {
        const dbl temp = m[k * n + i];
        m[k * n + i] = m[k * n + j];
        m[k * n + j] = temp;
    }
}

__global__ static void column_to_null_down(dbl * __restrict__ const m,
    const uint n, const uint k)
{
    const uint
        idxX = threadIdx.x + blockDim.x * blockIdx.x,
        idxY = threadIdx.y + blockDim.y * blockIdx.y,
        offsetX = blockDim.x * gridDim.x,
        offsetY = blockDim.y * gridDim.y;
    const dbl m_k_k = m[k * n + k];
    for (uint j = k + 1 + idxY; j <= n; j += offsetY)
    {
        for (uint i = k + 1 + idxX; i < n; i += offsetX)
        {
            m[j * n + i] = fma(-m[k * n + i] / m_k_k,
                m[j * n + k], m[j * n + i]);
        }
    }
}

__global__ static void column_to_null_up(dbl * __restrict__ const m,
    const uint n, const uint k)
{
    const uint idx = threadIdx.x + blockDim.x * blockIdx.x,
        offset = blockDim.x * gridDim.x;
    const dbl m_k_k = m[k * n + k], m_k_n = m[n * n + k];
    for (uint i = idx; i < k; i += offset)
    {
        m[n * n + i] = fma(-m[k * n + i] / m_k_k, m_k_n, m[n * n + i]);
    }
}

__global__ static void solve(dbl * __restrict__ const m,
    const uint n)
{
    const uint idx = threadIdx.x + blockDim.x * blockIdx.x,
        offset = blockDim.x * gridDim.x;
    for (uint k = idx; k < n; k += offset)
    {
        m[n * n + k] /= m[k * n + k];
    }
}

__host__ static uint matrix_in(dbl ** const __restrict__ matrix_ptr)
{
    uint n;
    scanf("%u", &n);
    dbl * const matrix = (dbl *) malloc(sizeof(dbl) * n * (n + 1));
    for (uint i = 0; i < n; ++i)
    {
        for (uint j = 0; j < n; ++j)
        {
            scanf("%lf", matrix + j * n + i);
        }
    }
    for (uint i = 0; i < n; ++i)
    {
        scanf("%lf", matrix + n * n + i);
    }

    *matrix_ptr = matrix;

    return n;
}

__host__ static void matrix_out(dbl * __restrict__ const matrix, const uint n)
{
    for (uint i = 0; i < n; ++i)
    {
        printf("%.10le ", matrix[n * n + i]);
    }
    putchar('\n');

    free(matrix);
}
