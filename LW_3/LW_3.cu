#include <assert.h>
#include <errno.h>
#include <float.h>
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

#define max(a, b) ((a) >= (b) ? (a) : (b))
#define min(a, b) ((a) <= (b) ? (a) : (b))

#define swap(a, b, temp) \
    do { \
        (temp) = (a); \
        (a) = (b); \
        (b) = (temp); \
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

#define MU_MAX (32U)
/*__device__ ?*/
__device__ __constant__ float3 dev_mu[MU_MAX];

__host__ void k_means(uchar4 * __restrict__, uint, uint,
    float3 * __restrict__, uint);
__global__ void assignment_step(uchar4 * __restrict__, uint, uint, uint);
__host__ bool update_step(const uchar4 * __restrict__,
    const uchar4 * __restrict__, uint, uint, float3 * __restrict__, uint);

__host__ uchar4 *file_in_and_malloc(const char * __restrict__,
    uint * __restrict__, uint * __restrict__);
__host__ void file_out_and_free(const char * __restrict__,
    uchar4 * __restrict__, uint, uint);

int main(void)
{
    char filename_in[FILENAME_MAX], filename_out[FILENAME_MAX];
    uint k;
    scanf("%s%s%u", filename_in, filename_out, &k);

    uint w, h;
    uchar4 * const data = file_in_and_malloc(filename_in, &w, &h);

    float3 init_mu[MU_MAX];
    for (uint i = 0; i < k; ++i)
    {
        uint x, y;
        scanf("%u%u", &x, &y);
        init_mu[i].x = data[y * w + x].x;
        init_mu[i].y = data[y * w + x].y;
        init_mu[i].z = data[y * w + x].z;
    }

    k_means(data, w, h, init_mu, k);

    file_out_and_free(filename_out, data, w, h);

    return 0;
}

__host__ void k_means(uchar4 * __restrict__ data_before,
    const uint w, const uint h, float3 * const __restrict__ mu, const uint k)
{
    uchar4 * __restrict__ data = (uchar4 *) malloc(sizeof(uchar4) * w * h);
    exit_if(!data, "malloc()");

    uchar4 *dev_data, *data_temp;
    cudaErrorCheck(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));
    cudaErrorCheck(cudaMemcpy(dev_data, data_before, sizeof(uchar4) * w * h,
        cudaMemcpyHostToDevice));

    bool flag = true, flag_data = true;
    while (flag)
    {
        cudaMemcpyToSymbol(dev_mu, mu, sizeof(float3) * k);
        assignment_step<<<dim3(32U, 32U), dim3(32U, 32U)>>>(dev_data, w, h, k);
        cudaErrorCheck(cudaGetLastError());
        cudaErrorCheck(cudaMemcpy(data, dev_data, sizeof(uchar4) * w * h,
            cudaMemcpyDeviceToHost));

        flag = update_step(data, data_before, w, h, mu, k);

        swap(data, data_before, data_temp);
        flag_data = !flag_data;
    }

    cudaErrorCheck(cudaFree(dev_data));

    if (!flag_data)
    {
        swap(data, data_before, data_temp);
        memcpy(data_before, data, sizeof(uchar4) * w * h);
    }

    free(data);
}

__global__ void assignment_step(uchar4 * const __restrict__ dev_data,
    const uint w, const uint h, const uint k)
{
    const uint
        idxX = threadIdx.x + blockDim.x * blockIdx.x,
        idxY = threadIdx.y + blockDim.y * blockIdx.y,
        offsetX = blockDim.x * gridDim.x,
        offsetY = blockDim.y * gridDim.y;
    for (uint i = idxX; i < w; i += offsetX)
    {
        for (uint j = idxY; j < h; j += offsetY)
        {
            flt min = FLT_MAX;
            const uchar4 cur_data = dev_data[j * w + i];
            for (uint idx = 0; idx < k; ++idx)
            {
                const flt cur = sqrtf(
                    (cur_data.x - dev_mu[idx].x) * (cur_data.x - dev_mu[idx].x) +
                    (cur_data.y - dev_mu[idx].y) * (cur_data.y - dev_mu[idx].y) +
                    (cur_data.z - dev_mu[idx].z) * (cur_data.z - dev_mu[idx].z));
                if (cur < min)
                {
                    dev_data[j * w + i].w = idx;
                    min = cur;
                }
            }
        }
    }
}

__host__ bool update_step(const uchar4 * const __restrict__ data,
    const uchar4 * const __restrict__ data_before, const uint w, const uint h,
    float3 * const __restrict__ mu, const uint k)
{
    ulonglong3 accumulate[MU_MAX] = { make_ulonglong3(0, 0, 0) };
    uint count[MU_MAX] = { 0 };
    bool flag = false;
    for (uint i = 0; i < w * h; ++i)
    {
        accumulate[data[i].w].x += data[i].x;
        accumulate[data[i].w].y += data[i].y;
        accumulate[data[i].w].z += data[i].z;
        ++count[data[i].w];
        if (data[i].w != data_before[i].w)
        {
            flag = true;
        }
    }
    for (uint i = 0; i < k; ++i)
    {
        mu[i].x = (flt) accumulate[i].x / count[i];
        mu[i].y = (flt) accumulate[i].y / count[i];
        mu[i].z = (flt) accumulate[i].z / count[i];
    }

    return flag;
}

__host__ uchar4 *file_in_and_malloc(const char * const __restrict__ filename,
    uint * const __restrict__ w_ptr, uint * const __restrict__ h_ptr)
{
    FILE * const f_in = fopen(filename, "rb");
    exit_if(!f_in, "fopen()");

    uint w, h;
    fread(&w, sizeof(uint), 1, f_in);
    fread(&h, sizeof(uint), 1, f_in);

    uchar4 * const data = (uchar4 *) malloc(sizeof(uchar4) * w * h);
    exit_if(!data, "malloc()");

    fread(data, sizeof(uchar4), w * h, f_in);
    fclose(f_in);

    *w_ptr = w;
    *h_ptr = h;
    return data;
}

__host__ void file_out_and_free(const char * const __restrict__ filename,
    uchar4 * const __restrict__ data, const uint w, const uint h)
{
    FILE * const f_out = fopen(filename, "wb");
    exit_if(!f_out, "fopen()");

    fwrite(&w, sizeof(uint), 1, f_out);
    fwrite(&h, sizeof(uint), 1, f_out);
    fwrite(data, sizeof(uchar4), w * h, f_out);
    fclose(f_out);

    free(data);
}
