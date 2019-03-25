#include <assert.h>
#include <errno.h>
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

void minus(dbl * const restrict first,
    const dbl * const restrict second,
    const size_t n)
{
    size_t i = 0;
    while (i < n)
    {
        first[i] -= second[i];
        ++i;
    }
}

int main(void)
{
    size_t n;
    scanf("%zu", &n);
    dbl * const first  = (dbl *) malloc(sizeof(dbl) * n),
        * const second = (dbl *) malloc(sizeof(dbl) * n);
    exit_if(!first || !second, "malloc()");
    memset(first, 0, n * sizeof(dbl));
    memset(second, 0, n * sizeof(dbl));

    clock_t start = clock(), stop;
    minus(first, second, n);
    stop = clock();
    printf("%lf\n", (dbl) (stop - start) / CLOCKS_PER_SEC);
    free(first);
    free(second);

    return 0;
}
