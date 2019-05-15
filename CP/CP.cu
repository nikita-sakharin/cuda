#include <assert.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <GL/glew.h>

#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#include <curand.h>
#include <curand_kernel.h>

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
        cudaError_t result = error; \
        if (result != cudaSuccess) \
        { \
            fprintf(stderr, "cuda %s:%d error: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(result)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define curandErrorCheck(status) \
    do { \
        curandStatus_t result = status; \
        if (result != CURAND_STATUS_SUCCESS) \
        { \
            fprintf(stderr, "curand %s:%d error: %d\n", __FILE__, __LINE__, \
                result); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define WIDTH (1280U)
#define HEIGHT (720U)

#define HALF_POW (2.0F)
#define EPSILON (4.92156660115E-3F)
#define W (9.95078433398E-1F)
#define A_1 (0.25F)
#define A_2 (0.75F)
#define G (1.0F)
#define DELTA_T (4.92156660115E-3F)

#define STEP (1.125F)

#define N (4096U)

typedef struct
{
    float2 center,
           scale;
} LinearMap;

class Compare
{
public:
    __device__ bool operator()(const float2 &, const float2 &) const;
};

class Reduce
{
public:
    __device__ float2 operator()(const float2 &, const float2 &) const;
};

static void display(void);
static void keyboard(uchar, int, int);
static void idle(void);

__global__ static void initPoint(float2 * __restrict__, float2 * __restrict__,
    curandState_t * __restrict__, uint, float2, float2, ullong);

__global__ static void updatePoint(float2 * __restrict__,
    float2 * __restrict__, float2 * __restrict__, float2 * __restrict__,
    curandState_t * __restrict__, uint);
__global__ static void fillValue(flt * __restrict__);
__global__ static void fillColormap(uchar4 * __restrict__,
    const flt * __restrict__);
__global__ static void renderPoint(uchar4 * __restrict__,
    const float2 * __restrict__, uint, bool);

__host__ static void uniformPoint(float2 * __restrict__, float2 * __restrict__,
    float2 * __restrict__, uint, ullong);

__host__ static void gbest_update(const float2 * __restrict__, uint,
    const Compare &);
__host__ static void minmax_update(const flt * __restrict__);
__host__ static void map_update(void);
__host__ static float2 median_update(const float2 * __restrict__, uint,
    const Reduce &);

__device__ static void updateForce(const float2 * __restrict__,
    float2 * __restrict__, uint, uint);
__device__ static void updateVelocityPosition(float2 * __restrict__,
    float2 * __restrict__, const float2 * __restrict__,
    const float2 * __restrict__, curandState_t * __restrict__, uint, uint);
__device__ static flt himmelblauMap(uint, uint);
__device__ static flt himmelblau(flt, flt);
__device__ static uchar4 colormap(flt);
__device__ static void drawPixels(uchar4 * __restrict__, bool, int, int);

static int window;

static struct cudaGraphicsResource *resource;
static GLuint vbo;

static const dim3
    block1 = { 16U },
    thread1 = { 256U },
    block2 = { 4U, 4U },
    thread2 = { 16U, 16U };

static const float2
    UNIFORM_A = { -24.0F, -24.0F },
    UNIFORM_B = { 24.0F, 24.0F };

static float2 *pos_global;
static float2 *v_global;
static float2 *f_global;
static float2 *lbest_global;
static curandState_t *state_global;

static flt *value_global;

__device__ __constant__ static float2 gbest;
__device__ __constant__ static flt minValue, maxValue;
__device__ __constant__ static LinearMap map;

static LinearMap map_host = {
    { 0.0F, 0.0F }, { 128.0F, 128.0F / WIDTH * HEIGHT } };
static bool is_reset_position = true;

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutInitWindowPosition(-1, -1);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

    window = glutCreateWindow("Himmelblau's function");

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutIdleFunc(idle);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluOrtho2D(0.0, (GLdouble) WIDTH, 0.0, (GLdouble) HEIGHT);

    glewInit();

    glGenBuffers((GLsizei) 1, &vbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, WIDTH * HEIGHT * sizeof(uchar4),
        (GLvoid *) NULL, GL_DYNAMIC_DRAW);

    cudaErrorCheck(cudaGraphicsGLRegisterBuffer(&resource, vbo,
        cudaGraphicsMapFlagsWriteDiscard));

    cudaErrorCheck(cudaMalloc(&pos_global, N * sizeof(float2)));
    cudaErrorCheck(cudaMalloc(&v_global, N * sizeof(float2)));
    cudaErrorCheck(cudaMalloc(&f_global, N * sizeof(float2)));
    cudaErrorCheck(cudaMalloc(&lbest_global, N * sizeof(float2)));
    cudaErrorCheck(cudaMalloc(&state_global, N * sizeof(curandState_t)));
    cudaErrorCheck(cudaMalloc(&value_global, WIDTH * HEIGHT * sizeof(flt)));

    const ullong seed = time(NULL) * CLOCKS_PER_SEC + clock();
    printf("seed = %llu\n", seed);

    uniformPoint(pos_global, v_global, f_global, N, seed);
    initPoint<<<block1, thread1>>>(pos_global, lbest_global, state_global, N,
        UNIFORM_A, UNIFORM_B, seed);
    cudaErrorCheck(cudaGetLastError());

    glutMainLoop();

    return 0;
}

__device__ inline bool Compare::operator()(const float2 &a,
    const float2 &b) const
{
    return himmelblau(a.x, a.y) < himmelblau(b.x, b.y);
}

__device__ inline float2 Reduce::operator()(const float2 &a,
    const float2 &b) const
{
    return { a.x + b.x, a.y + b.y };
}

static void display(void)
{
    glClearColor(0.0F, 0.0F, 0.0F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glutSwapBuffers();
}

static void keyboard(const uchar key, const int x, const int y)
{
    switch (key)
    {
    case 27: // ESC
        cudaErrorCheck(cudaGraphicsUnregisterResource(resource));
        glDeleteBuffers(1, &vbo);
        glutDestroyWindow(window);
        cudaErrorCheck(cudaFree(pos_global));
        cudaErrorCheck(cudaFree(v_global));
        cudaErrorCheck(cudaFree(f_global));
        cudaErrorCheck(cudaFree(lbest_global));
        cudaErrorCheck(cudaFree(state_global));
        cudaErrorCheck(cudaFree(value_global));
        exit(EXIT_SUCCESS);
        break;

    case '+':
        map_host.scale.x /= STEP;
        break;

    case '-':
        map_host.scale.x *= STEP;
        break;

    default:
        return;
    }

    map_host.scale.y = map_host.scale.x / WIDTH * HEIGHT;
    map_update();
    is_reset_position = true;
}

static void idle(void)
{
    static const Compare compare;
    static const Reduce reduce;

    uchar4* data;
    size_t size;
    cudaErrorCheck(cudaGraphicsMapResources(1, &resource, 0));
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void **) &data, &size,
        resource));

    const float2 median = median_update(pos_global, N, reduce);

    if (fabsf(map_host.center.x - median.x) >= map_host.scale.x / 2.0F ||
        fabsf(map_host.center.y - median.y) >= map_host.scale.y / 2.0F ||
        is_reset_position)
    {
        map_host.center = median;
        map_update();
        is_reset_position = false;

        fillValue<<<block2, thread2>>>(value_global);
        cudaErrorCheck(cudaGetLastError());

        minmax_update(value_global);

        fillColormap<<<block2, thread2>>>(data, value_global);
        cudaErrorCheck(cudaGetLastError());
    }
    else
    {
        renderPoint<<<block1, thread1>>>(data, pos_global, N, false);
    }

    gbest_update(lbest_global, N, compare);
    updatePoint<<<block1, thread1>>>(pos_global, v_global, f_global,
        lbest_global, state_global, N);

    renderPoint<<<block1, thread1>>>(data, pos_global, N, true);

    cudaErrorCheck(cudaGraphicsUnmapResources(1, &resource, 0));
    glutPostRedisplay();
}

__global__ static void initPoint(float2 * __restrict__ const pos,
    float2 * __restrict__ const lbest, curandState_t * __restrict__ const state,
    const uint n, const float2 a, const float2 b, const ullong seed)
{
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x,
        offset = blockDim.x * gridDim.x;
    for (uint i = idx; i < n; i += offset)
    {
        pos[i].x = lbest[i].x = pos[i].x * (b.x - a.x) + a.x;
        pos[i].y = lbest[i].y = pos[i].y * (b.y - a.y) + a.y;
        curand_init(seed + i, seed, seed, state + i);
    }
}

__global__ static void updatePoint(float2 * __restrict__ const pos,
    float2 * __restrict__ const v, float2 * __restrict__ const f,
    float2 * __restrict__ const lbest, curandState_t * __restrict__ const state,
    const uint n)
{
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x,
        offset = blockDim.x * gridDim.x;

    for (uint i = idx; i < n; i += offset)
    {
        updateForce(pos, f, n, i);
        updateVelocityPosition(pos, v, f, lbest, state, n, i);
        if (himmelblau(pos[i].x, pos[i].y) <
            himmelblau(lbest[i].x, lbest[i].y))
        {
            lbest[i] = pos[i];
        }
    }
}

__global__ static void fillValue(flt * __restrict__ const value)
{
    const uint
        idxX = blockIdx.x * blockDim.x + threadIdx.x,
        idxY = blockIdx.y * blockDim.y + threadIdx.y,
        offsetX = blockDim.x * gridDim.x,
        offsetY = blockDim.y * gridDim.y;

    for (uint j = idxY; j < HEIGHT; j += offsetY)
    {
        for (uint i = idxX; i < WIDTH; i += offsetX)
        {
            value[j * WIDTH + i] = himmelblauMap(i, j);
        }
    }
}

__global__ static void fillColormap(uchar4 * __restrict__ const data,
    const flt * __restrict__ const value)
{
    const uint
        idxX = blockIdx.x * blockDim.x + threadIdx.x,
        idxY = blockIdx.y * blockDim.y + threadIdx.y,
        offsetX = blockDim.x * gridDim.x,
        offsetY = blockDim.y * gridDim.y;
    for (uint j = idxY; j < HEIGHT; j += offsetY)
    {
        for (uint i = idxX; i < WIDTH; i += offsetX)
        {
            data[j * WIDTH + i] = colormap((value[j * WIDTH + i] - minValue) /
                (maxValue - minValue));
        }
    }
}

__global__ static void renderPoint(uchar4 * __restrict__ const data,
    const float2 * __restrict__ const pos, const uint n, const bool mode)
{
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x,
        offset = blockDim.x * gridDim.x;
    for (uint k = idx; k < n; k += offset)
    {
        const float2 pos_k = pos[k];
        if (fabsf(pos_k.x - map.center.x) <= map.scale.x &&
            fabsf(pos_k.y - map.center.y) <= map.scale.y)
        {
            const int
                i = roundf((pos_k.x - map.center.x + map.scale.x) /
                    (2.0F * map.scale.x) * (WIDTH - 1)),
                j = roundf((pos_k.y - map.center.y + map.scale.y) /
                    (2.0F * map.scale.y) * (HEIGHT - 1));
            drawPixels(data, mode, i, j);
            drawPixels(data, mode, i - 1, j - 1);
            drawPixels(data, mode, i + 1, j - 1);
            drawPixels(data, mode, i - 1, j + 1);
            drawPixels(data, mode, i + 1, j + 1);
        }
    }
}

__host__ static void uniformPoint(float2 * __restrict__ const pos,
    float2 * __restrict__ const v, float2 * __restrict__ const f,
    const uint n, const ullong seed)
{
    assert(sizeof(float2) % sizeof(flt) == 0);
    curandGenerator_t generator;
    curandErrorCheck(curandCreateGenerator(&generator,
        CURAND_RNG_PSEUDO_DEFAULT));
    curandErrorCheck(curandSetPseudoRandomGeneratorSeed(generator, seed));

    curandErrorCheck(curandGenerateUniform(generator, (flt *) pos,
        n * sizeof(float2) / sizeof(flt)));

    cudaErrorCheck(cudaMemset(v, 0, n * sizeof(float2)));
    cudaErrorCheck(cudaMemset(f, 0, n * sizeof(float2)));

    curandErrorCheck(curandDestroyGenerator(generator));
}

__host__ static void gbest_update(const float2 * __restrict__ const lbest,
    const uint n, const Compare &compare)
{
    cudaErrorCheck(cudaMemcpyToSymbol(gbest,
        thrust::min_element(thrust::device, lbest, lbest + n, compare),
        sizeof(float2), 0, cudaMemcpyDeviceToDevice));
}

__host__ static void minmax_update(const flt * __restrict__ const value)
{
    const thrust::pair<const flt *, const flt *> minmax =
        thrust::minmax_element(thrust::device, value, value + WIDTH * HEIGHT);
    cudaErrorCheck(cudaMemcpyToSymbol(minValue, minmax.first, sizeof(flt), 0,
        cudaMemcpyDeviceToDevice));
    cudaErrorCheck(cudaMemcpyToSymbol(maxValue, minmax.second, sizeof(flt), 0,
        cudaMemcpyDeviceToDevice));
}

__host__ static void map_update(void)
{
    cudaErrorCheck(cudaMemcpyToSymbol(map, &map_host, sizeof(LinearMap), 0,
        cudaMemcpyHostToDevice));
}

__host__ static float2 median_update(const float2 * __restrict__ const pos,
    const uint n, const Reduce &reduce)
{
    static const float2 init = { 0.0, 0.0 };
    float2 median = thrust::reduce(thrust::device, pos, pos + n,
        init, reduce);
    median.x /= n;
    median.y /= n;
    return median;
}

__device__ static void updateForce(const float2 * __restrict__ const pos,
    float2 * __restrict__ const f, const uint n, const uint i)
{
    f[i].x = f[i].y = 0.0F;
    for (uint j = 0; j < n; ++j)
    {
        if (j == i)
        {
            continue;
        }
        const flt rho = powf(
            (pos[i].x - pos[j].x) * (pos[i].x - pos[j].x) +
            (pos[i].y - pos[j].y) * (pos[i].y - pos[j].y), HALF_POW) + EPSILON;

        f[i].x += (pos[i].x - pos[j].x) / rho;
        f[i].y += (pos[i].y - pos[j].y) / rho;
    }
}

__device__ static void updateVelocityPosition(float2 * __restrict__ const pos,
    float2 * __restrict__ const v, const float2 * __restrict__ const f,
    const float2 * __restrict__ const lbest,
    curandState_t * __restrict__ const state, const uint n, const uint i)
{
    const flt k_1 = curand_uniform(state + i), k_2 = curand_uniform(state + i);

    v[i].x = W * v[i].x + (A_1 * k_1 * (lbest[i].x - pos[i].x) +
        A_2 * k_2 * (gbest.x - pos[i].x) + G * f[i].x) * DELTA_T;
    v[i].y = W * v[i].y + (A_1 * k_1 * (lbest[i].y - pos[i].y) +
        A_2 * k_2 * (gbest.y - pos[i].y) + G * f[i].y) * DELTA_T;

    pos[i].x = pos[i].x + v[i].x * DELTA_T;
    pos[i].y = pos[i].y + v[i].y * DELTA_T;
}

__device__ inline static flt himmelblauMap(const uint i, const uint j)
{
    const flt x = 2.0F * i / (WIDTH - 1) - 1.0F,
              y = 2.0F * j / (HEIGHT - 1) - 1.0F;
    return himmelblau(map.scale.x * x + map.center.x,
        map.scale.y * y + map.center.y);
}

__device__ inline static flt himmelblau(const flt x, const flt y)
{
    return (x * x + y - 11.0F) * (x * x + y - 11.0F) +
        (x + y * y - 7.0F) * (x + y * y - 7.0F);
}

__device__ __constant__ static const uint size = 6U;
__device__ __constant__ static const flt x[] = {
    0.0F, 0.125F, 0.375F, 0.625F, 0.875F, 1.0F };

__device__ __constant__ static const uchar4 jet[] = {
    { 0, 0, UCHAR_MAX / 2, UCHAR_MAX }, { 0, 0, UCHAR_MAX, UCHAR_MAX },
    { 0, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX },
    { UCHAR_MAX, UCHAR_MAX, 0, UCHAR_MAX },
    { UCHAR_MAX, 0, 0, UCHAR_MAX }, { UCHAR_MAX / 2, 0, 0, UCHAR_MAX } };

__device__ static uchar4 colormap(const flt value)
{
    assert(-FLT_EPSILON <= value && value <= 1.0F + FLT_EPSILON);

    if (value < x[0])
    {
        return jet[0];
    }
    for (uint i = 1; i < size; ++i)
    {
        if (value <= x[i])
        {
            const flt x0 = x[i] - value, x1 = (value - x[i - 1]),
                deltaX = x[i] - x[i - 1];
            return (uchar4) {
                roundf((jet[i - 1].x * x0 + jet[i].x * x1) / deltaX),
                roundf((jet[i - 1].y * x0 + jet[i].y * x1) / deltaX),
                roundf((jet[i - 1].z * x0 + jet[i].z * x1) / deltaX),
                UCHAR_MAX
            };
        }
    }

    return jet[size - 1];
}

__device__ inline static void drawPixels(uchar4 * __restrict__ const data,
    const bool mode, const int i, const int j)
{
    if (i < 0 || i >= WIDTH || j < 0 || j >= HEIGHT)
    {
        return;
    }

    data[j * WIDTH + i] = (mode ?
        (uchar4) { 0, 0, 0, UCHAR_MAX } :
        colormap((himmelblauMap(i, j) - minValue) / (maxValue - minValue)));
}
