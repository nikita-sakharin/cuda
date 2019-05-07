#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

#define exit_if(cnd_value, msg) \
    do { \
        if (cnd_value) \
        { \
            if (errno) \
                perror(msg); \
            else \
                fprintf(stderr, "error: %s\n", msg); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define cudaErrorCheck(error) \
    do { \
        cudaError_t res = error; \
        if (res != cudaSuccess) \
        { \
            fprintf(stderr, "cuda %s:%d error: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(res)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define max(a, b) ((a) >= (b) ? (a) : (b))
#define min(a, b) ((a) <= (b) ? (a) : (b))

typedef enum
{
    ASC = 1,
    DESC = -1
} Order;

#define BLOCK_DIM_GLOBAL (256U)
#define  GRID_DIM_GLOBAL (32U)

#define BLOCK_DIM_SHARED (1024U)
#define  GRID_DIM_SHARED (16U)

#define SHARED_DIM (BLOCK_DIM_SHARED * 2U)

__host__ static uint greater_or_equal_pow_two(uint);
__host__ __device__ static void compare_swap(int * __restrict__,
    int * __restrict__, Order);

__host__ static void sort(int * __restrict__, uint, Order);

__global__ static void sort_global(int * __restrict__, uint, uint,
    uint, uint, Order);
__global__ static void sort_shared(int * __restrict__, uint, uint, Order);

__host__ static uint data_in(int ** __restrict__, uint * __restrict__);
__host__ static void data_out(int * __restrict__, uint);


int main(void)
{
    int *data;
    uint align_n;
    const uint n = data_in(&data, &align_n);

    sort(data, align_n, ASC);

    data_out(data, n);

    return 0;
}

__host__ inline static uint greater_or_equal_pow_two(uint value)
{
    --value;
    value |= value >> 1;
    value |= value >> 2U;
    value |= value >> 4U;
    value |= value >> 8U;
    value |= value >> 16U;
    return ++value;
}

__host__ __device__ inline static void compare_swap(int * __restrict__ const a,
    int * __restrict__ const b, const Order order)
{
    const int compare = (*a > *b) - (*a < *b);
    if (order == compare)
    {
        const int temp = *a;
        *a = *b;
        *b = temp;
    }
}

__host__ static void sort(int * __restrict__ const data, const uint size,
    const Order order)
{
    int *device_data;
    cudaErrorCheck(cudaMalloc(&device_data, sizeof(int) * size));
    cudaErrorCheck(cudaMemcpy(device_data, data, sizeof(int) * size,
        cudaMemcpyHostToDevice));
    for (uint n = 2U, shift_n = 1; n <= size; n <<= 1, ++shift_n)
    {
        for (uint half_k = n >> 1, shift_half_k = shift_n - 1; half_k;
            half_k >>= 1, --shift_half_k)
        {

            if (half_k == SHARED_DIM >> 1)
            {
                sort_shared<<<GRID_DIM_SHARED, BLOCK_DIM_SHARED>>>
                    (device_data, size, shift_n, order);
                break;
            }

            sort_global<<<GRID_DIM_GLOBAL, BLOCK_DIM_GLOBAL>>>(device_data, size,
                shift_n, half_k, shift_half_k, order);
            cudaErrorCheck(cudaGetLastError());
        }
    }

    cudaErrorCheck(cudaMemcpy(data, device_data, sizeof(int) * size,
        cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaFree(device_data));
}

__global__ static void sort_global(int * __restrict__ const data, const uint size,
    const uint shift_n, const uint half_k, const uint shift_half_k, Order order)
{
    const uint idx = threadIdx.x + blockDim.x * blockIdx.x,
        offset = blockDim.x * gridDim.x,
        increment = max(half_k, offset);

    uint i = (idx << 1) - (idx & half_k - 1), i_before;
    if (i >> shift_n & 1)
    {
        order = (Order) -order;
    }

    while (i < size)
    {
        compare_swap(data + i, data + half_k + i, order);

        i_before = i;
        i += offset;
        if (i_before >> shift_half_k != i >> shift_half_k)
        {
            i += increment;
            if (i_before >> shift_n != i >> shift_n)
            {
                order = (Order) -order;
            }
        }
    }
}

__global__ static void sort_shared(int * __restrict__ const data, const uint size,
    const uint shift_n, Order order)
{
    __shared__ int shared[SHARED_DIM];
    for (uint i = blockIdx.x * SHARED_DIM, i_before = i - (i & 1 << shift_n);
        i < size; i_before = i, i += SHARED_DIM * gridDim.x)
    {
        if (i_before >> shift_n != i >> shift_n)
        {
            order = (Order) -order;
        }

        shared[threadIdx.x] = data[i + threadIdx.x];
        shared[BLOCK_DIM_SHARED + threadIdx.x] =
            data[i + BLOCK_DIM_SHARED + threadIdx.x];
        __syncthreads();

        for (uint n = SHARED_DIM; n > 1; n >>= 1)
        {
            for (uint half_k = n >> 1; half_k; half_k >>= 1)
            {
                const uint j = (threadIdx.x << 1) - (threadIdx.x & half_k - 1);
                compare_swap(shared + j, shared + half_k + j, order);
                __syncthreads();
            }
        }

        data[i + threadIdx.x] = shared[threadIdx.x];
        data[i + BLOCK_DIM_SHARED + threadIdx.x] =
            shared[BLOCK_DIM_SHARED + threadIdx.x];
    }
}

__host__ static uint data_in(int ** __restrict__ const data_ptr,
    uint * __restrict__ const align_n_ptr)
{
    uint n;
    fread(&n, sizeof(uint), 1, stdin);
    const uint align_n = greater_or_equal_pow_two(n);

    int * const data = (int *) malloc(sizeof(int) * align_n);
    exit_if(!data, "malloc()");

    fread(data, sizeof(int), n, stdin);
    for (uint i = n; i < align_n; ++i)
    {
        data[i] = INT_MAX;
    }

    *data_ptr = data;
    *align_n_ptr = align_n;

    return n;
}

__host__ static void data_out(int * __restrict__ const data, const uint n)
{
    fwrite(data, sizeof(int), n, stdout);
    free(data);
}
