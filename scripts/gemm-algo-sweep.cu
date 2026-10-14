// P3 probe: does any cublasGemmEx algorithm beat CUBLAS_GEMM_DEFAULT_TENSOR_OP (what ggml uses)
// at the prefill GEMM shapes, on one P100? Same call as ggml: OP_T/OP_N, F16 in/out, COMPUTE_16F.
// Build: nvcc -O2 -arch=sm_60 -o gemm-algo-sweep scripts/gemm-algo-sweep.cu -lcublas
// Run:   CUDA_VISIBLE_DEVICES=0 ./gemm-algo-sweep   (prints CSV: shape,algo,ms,tflops)
// Confirm mode: ./gemm-algo-sweep -1,3 6  -> only those algos (COMPUTE_16F), interleaved over 6 rounds,
// so clock drift under the power cap hits every algorithm equally. Third arg: N (default 2048).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CK(x) do { auto e = (x); if (e != 0) { printf("error %d at %s:%d\n", (int) e, __FILE__, __LINE__); return 1; } } while (0)

int main(int argc, char ** argv) {
    std::vector<int> algos;
    if (argc > 1) { for (char * p = argv[1]; *p; ) { algos.push_back(strtol(p, &p, 10)); if (*p == ',') ++p; } }
    else { for (int a = -1; a <= 23; ++a) algos.push_back(a); }
    const int rounds = argc > 2 ? atoi(argv[2]) : 1;
    const int ncompute = argc > 1 ? 1 : 2;
    const int N = argc > 3 ? atoi(argv[3]) : 2048;  // ubatch (tokens)
    // {M = ne01 (output rows), K = ne00}: per-card shapes under -sm tensor (2 cards), then full (1 card)
    const int shapes[][2] = {
        {8704, 5120}, {5120, 8704}, {6144, 5120}, {5120, 6144}, {3072, 5120}, {5120, 3072},
        {17408, 5120}, {5120, 17408}, {12288, 5120}, {10240, 5120},
    };
    const size_t max_elems = 17408ull * 5120;
    half *A, *B, *C;
    CK(cudaMalloc(&A, max_elems * sizeof(half)));
    CK(cudaMalloc(&B, size_t(17408) * N * sizeof(half)));
    CK(cudaMalloc(&C, size_t(17408) * N * sizeof(half)));
    CK(cudaMemset(A, 0x11, max_elems * sizeof(half)));  // small finite halves
    CK(cudaMemset(B, 0x11, size_t(17408) * N * sizeof(half)));
    cublasHandle_t h;
    CK(cublasCreate(&h));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    const float alpha32 = 1.0f, beta32 = 0.0f;

    printf("round,M,K,algo,compute,ms,tflops\n");
    for (int round = 0; round < rounds; ++round)
    for (auto & s : shapes) {
        const int M = s[0], K = s[1];
        for (int compute = 0; compute < ncompute; ++compute) {          // 0: COMPUTE_16F (ggml), 1: COMPUTE_32F
            for (int algo : algos) {
                const cublasGemmAlgo_t a = algo < 0 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : (cublasGemmAlgo_t) algo;
                auto run = [&]() {
                    return compute == 0
                        ? cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_16F, K, B, CUDA_R_16F, K,
                                       &beta, C, CUDA_R_16F, M, CUBLAS_COMPUTE_16F, a)
                        : cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha32, A, CUDA_R_16F, K, B, CUDA_R_16F, K,
                                       &beta32, C, CUDA_R_16F, M, CUBLAS_COMPUTE_32F, a);
                };
                if (run() != CUBLAS_STATUS_SUCCESS) {
                    continue;  // algo not supported for this config
                }
                run();
                cudaEventRecord(e0);
                const int iters = 5;
                for (int i = 0; i < iters; ++i) run();
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= iters;
                printf("%d,%d,%d,%d,%s,%.3f,%.2f\n", round, M, K, algo, compute ? "32F" : "16F", ms, 2.0 * M * N * K / ms / 1e9);
            }
        }
        fflush(stdout);
    }
    return 0;
}
