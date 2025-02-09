#include <iostream>
#include <chrono>  // Include the chrono library
#include <random>  // for random number generators
#include <thread>  //for thread
#include <immintrin.h> //for avx

void sinx(int N,int terms,float* x,float* y){
    for(int i=0;i<N;i++){
        float value=x[i];
        float numer = x[i] * x[i] * x[i];
        int denom = 6; // 3!
        int sign = -1;
        for (int j=1; j<=terms; j++)
        {
            value += sign * numer / denom;
            numer *= x[i] * x[i];
            denom *= (2*j+2) * (2*j+3);
            sign *= -1;
        }
        y[i] = value;

    }
}

typedef struct {
    int N;
    int terms;
    float* x;
    float* y;
} my_args;
void my_thread_func(my_args* args)
{
    sinx(args->N, args->terms, args->x, args->y); // do work
}
void parallel_sinx(int N, int terms, float* x, float* y)
{
    std::thread my_thread;
    my_args args;
    args.N = N/2;
    args.terms = terms;
    args.x = x;
    args.y = y;
    my_thread = std::thread(my_thread_func, &args); // launch thread
    sinx(N - args.N, terms, x + args.N, y + args.N); // do work on main thread
    my_thread.join(); // wait for thread to complete
}
void simd_sinx(int N, int terms, float* x, float* y) {
    for (int i=0; i<N; i+=8)
    {
        __m256 origx = _mm256_load_ps(&x[i]);
        __m256 value = origx;
        __m256 numer = _mm256_mul_ps(origx, _mm256_mul_ps(origx, origx));
        __m256 denom = _mm256_set1_ps(6.0f); // 3!
        __m256 sign_vec = _mm256_set1_ps(-1.0f);

        for (int j = 1; j <= terms; j++) {
            __m256 tmp = _mm256_div_ps(_mm256_mul_ps(sign_vec, numer), denom);
            value = _mm256_add_ps(value, tmp);
            numer = _mm256_mul_ps(numer, _mm256_mul_ps(origx, origx));
            float temp = static_cast<float>((2 * j + 2) * (2 * j + 3));
            denom = _mm256_mul_ps(denom, _mm256_broadcast_ss(&temp));
            sign_vec = _mm256_mul_ps(sign_vec, _mm256_set1_ps(-1.0f));
        }
        _mm256_store_ps(&y[i], value);
    }
}
int main() {
    int N=100000;
    int terms=20;
    float x[N];
    float y[N];
    std::cout << "Program N: " << N << std::endl;
    // Random number engine
    std::mt19937 rng(std::random_device{}());
    for (int i = 0; i < N; ++i) {
        std::uniform_real_distribution<float> dist(0.0, 1.0);
        float random_num = dist(rng);
        x[i]=random_num;
    }
    // navie sinx
    auto start = std::chrono::high_resolution_clock::now();
    sinx(N,terms,x,y);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Program execution time: " << duration.count() << " microseconds" << std::endl;
    // thread sinx
    auto start2 = std::chrono::high_resolution_clock::now();
    parallel_sinx(N,terms,x,y);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    std::cout << "Program execution time: " << duration2.count() << " microseconds" << std::endl;
    // simd sinx
    auto start3 = std::chrono::high_resolution_clock::now();
    simd_sinx(N,terms,x,y);
    auto end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);
    std::cout << "Program execution time: " << duration3.count() << " microseconds" << std::endl;

    return 0;
}
