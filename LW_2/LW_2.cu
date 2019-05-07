#include <errno.h>
#include <float.h>
#include <math.h>
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


#define DBL_CBRT_EPSILON (6.055454452393339060789E-6)

texture<uchar4, cudaTextureType2D, cudaReadModeElementType> tex;

__global__ void kernel(const uint tex_w, const uint tex_h,
    uchar4 * const __restrict__ dest, const uint dest_w, const uint dest_h)
{
    const int
        idxX = threadIdx.x + blockDim.x * blockIdx.x,
        idxY = threadIdx.y + blockDim.y * blockIdx.y,
        offsetX = blockDim.x * gridDim.x,
        offsetY = blockDim.y * gridDim.y;
    for (int i = idxX; i < dest_w; i += offsetX)
    {
        for (int j = idxY; j < dest_h; j += offsetY)
        {
            const dbl
                x = (i + 0.5) * tex_w / dest_w,
                y = (j + 0.5) * tex_h / dest_h;
            dbl
                x1 = floor(x - 0.5),
                x2 = ceil(x - 0.5),
                y1 = floor(y - 0.5),
                y2 = ceil(y - 0.5);
            const uchar4
                f11 = tex2D(tex, x1, y1),
                f12 = tex2D(tex, x1, y2),
                f21 = tex2D(tex, x2, y1),
                f22 = tex2D(tex, x2, y2);
            x1 += 0.5 - DBL_CBRT_EPSILON;
            x2 += 0.5 + DBL_CBRT_EPSILON;
            y1 += 0.5 - DBL_CBRT_EPSILON;
            y2 += 0.5 + DBL_CBRT_EPSILON;
            const dbl divides = (x2 - x1) * (y2 - y1);
            double4 f;
            f.x =
                f11.x * (x2 - x) * (y2 - y) +
                f12.x * (x2 - x) * (y - y1) +
                f21.x * (x - x1) * (y2 - y) +
                f22.x * (x - x1) * (y - y1);
            f.x /= divides;
            f.y =
                f11.y * (x2 - x) * (y2 - y) +
                f12.y * (x2 - x) * (y - y1) +
                f21.y * (x - x1) * (y2 - y) +
                f22.y * (x - x1) * (y - y1);
            f.y /= divides;
            f.z =
                f11.z * (x2 - x) * (y2 - y) +
                f12.z * (x2 - x) * (y - y1) +
                f21.z * (x - x1) * (y2 - y) +
                f22.z * (x - x1) * (y - y1);
            f.z /= divides;
            f.w =
                f11.w * (x2 - x) * (y2 - y) +
                f12.w * (x2 - x) * (y - y1) +
                f21.w * (x - x1) * (y2 - y) +
                f22.w * (x - x1) * (y - y1);
            f.w /= divides;
            dest[j * dest_w + i] = make_uchar4(f.x, f.y, f.z, f.w);
        }
    }
}

int main(void)
{
    char filename_in[FILENAME_MAX], filename_out[FILENAME_MAX];
    uint new_w, new_h;
    scanf("%s%s%u%u", filename_in, filename_out, &new_w, &new_h);

    FILE * const f_in = fopen(filename_in, "rb");
    exit_if(!f_in, "fopen()");

    uint w, h;
    fread(&w, sizeof(uint), 1, f_in);
    fread(&h, sizeof(uint), 1, f_in);

    uchar4 * const img = (uchar4 *) malloc(sizeof(uchar4) * max(w * h, new_w * new_h));
    exit_if(!img, "malloc()");
    fread(img, sizeof(uchar4), h * w, f_in);
    fclose(f_in);

    cudaChannelFormatDesc channel = cudaCreateChannelDesc<uchar4>();
    cudaErrorCheck(cudaGetLastError());

    cudaArray *device_array;
    cudaErrorCheck(cudaMallocArray(&device_array, &channel, w, h));
    cudaErrorCheck(cudaMemcpyToArray(device_array, 0, 0, img, sizeof(uchar4) * w * h,
        cudaMemcpyHostToDevice));

    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.channelDesc = channel;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = false;
    cudaErrorCheck(cudaBindTextureToArray(tex, device_array, channel));

    uchar4 *dev_img;
    cudaErrorCheck(cudaMalloc(&dev_img, sizeof(uchar4) * new_w * new_h));
    kernel<<<dim3(32U, 32U), dim3(32U, 32U)>>>(w, h, dev_img, new_w, new_h);
    cudaErrorCheck(cudaGetLastError());
    cudaErrorCheck(cudaMemcpy(img, dev_img, sizeof(uchar4) * new_w * new_h,
        cudaMemcpyDeviceToHost));

    FILE * const f_out = fopen(filename_out, "wb");
    exit_if(!f_out, "fopen()");

    fwrite(&new_w, sizeof(uint), 1, f_out);
    fwrite(&new_h, sizeof(uint), 1, f_out);
    fwrite(img, sizeof(uchar4), new_w * new_h, f_out);
    fclose(f_out);

    cudaErrorCheck(cudaUnbindTexture(tex));
    cudaErrorCheck(cudaFreeArray(device_array));
    cudaErrorCheck(cudaFree(dev_img));
    free(img);

    return 0;
}
