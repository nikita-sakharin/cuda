#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

typedef struct
{
    uchar x,
          y,
          z,
          w;
} uchar4;

typedef struct
{
    dbl x,
        y,
        z,
        w;
} double4;

static uchar4 make_uchar4(uchar, uchar, uchar, uchar);

#define DBL_CBRT_EPSILON (6.055454452393339060789E-6)

inline static int idx(const int i, const uint n)
{
    if (i < 0)
    {
        return 0;
    }
    else if (i >= n)
    {
        return n - 1;
    }
    return i;
}

static void kernel(uchar4 * const restrict src,
    const uint src_w, const uint src_h,
    uchar4 * const restrict dest, const uint dest_w, const uint dest_h)
{
    for (int i = 0; i < dest_w; ++i)
    {
        for (int j = 0; j < dest_h; ++j)
        {
            const dbl
                x = (i + 0.5) * src_w / dest_w,
                y = (j + 0.5) * src_h / dest_h;
            dbl
                x1 = idx(floor(x - 0.5), src_w),
                x2 = idx(ceil(x - 0.5), src_w),
                y1 = idx(floor(y - 0.5), src_h),
                y2 = idx(ceil(y - 0.5), src_h);
            const uchar4
                f11 = src[(int) (x1 + y1 * src_w)],
                f12 = src[(int) (x1 + y2 * src_w)],
                f21 = src[(int) (x2 + y1 * src_w)],
                f22 = src[(int) (x2 + y2 * src_w)];
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
    uint src_w, src_h, dest_w, dest_h;
    scanf("%u%u%u%u", &src_w, &src_h, &dest_w, &dest_h);

    uchar4 * const src  = (uchar4 *) malloc(sizeof(uchar4) * src_w * src_h),
           * const dest = (uchar4 *) malloc(sizeof(uchar4) * dest_w * dest_h);
    exit_if(!src || !dest, "malloc()");

    clock_t start = clock(), stop;

    kernel(src, src_w, src_h, dest, dest_w, dest_h);

    stop = clock();
    printf("%lf\n", (dbl) (stop - start) / CLOCKS_PER_SEC * 1000.0);

    free(dest);
    free(src);

    return 0;
}

inline static uchar4 make_uchar4(const uchar x, const uchar y, const uchar z,
    const uchar w)
{
    return (uchar4) { .x = x, .y = y, .z = z, .w = w };
}
