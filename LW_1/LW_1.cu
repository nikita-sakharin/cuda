#include <assert.h>
#include <errno.h>
#include <stdbool.h>
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

#define NUM_BLOCKS (256)
#define BLOCK_SIZE (256)

__global__ void kernel(dbl * const __restrict__ first,
    const dbl * const __restrict__ second,
    const size_t n)
{
    const size_t offset = gridDim.x * blockDim.x;
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    while (idx < n)
    {
        first[idx] -= second[idx];
        idx += offset;
    }
}

int main(void)
{
    size_t n;
    scanf("%zu", &n);
    dbl * const first  = (dbl *) malloc(sizeof(dbl) * n),
        * const second = (dbl *) malloc(sizeof(dbl) * n);
    exit_if(!first || !second, "malloc()");
    for(size_t i = 0; i < n; ++i)
    {
        scanf("%lf", first + i);
    }
    for(size_t i = 0; i < n; ++i)
    {
        scanf("%lf", second + i);
    }

    dbl *device_first, *device_second;
    cudaErrorCheck(cudaMalloc(&device_first, sizeof(dbl) * n));
    cudaErrorCheck(cudaMemcpy(device_first, first, sizeof(dbl) * n,
        cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMalloc(&device_second, sizeof(dbl) * n));
    cudaErrorCheck(cudaMemcpy(device_second, second, sizeof(dbl) * n,
        cudaMemcpyHostToDevice));

    kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(device_first, device_second, n);
    cudaErrorCheck(cudaGetLastError());

    cudaErrorCheck(cudaMemcpy(first, device_first, sizeof(dbl) * n,
        cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; ++i)
        printf("%.10e ", first[i]);
    putchar('\n');

    cudaErrorCheck(cudaFree(device_first));
    cudaErrorCheck(cudaFree(device_second));

    free(first);
    free(second);

    return 0;
}
