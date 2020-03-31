// GPU prefix sum reduce-and-sweep algorithm
//   Reduce: build a binary tree
//   Sweep: sweep the tree to get prefix sum
#include <cstdio>
#include <cuda.h>
#include "helper.h"

__global__ void rs_allshare(int N, ValueType* idata, ValueType* odata) {
    extern __shared__ ValueType temp[]; // allocated on invocation
    int I = threadIdx.x;
    // printf("%d %d %d\n", I, I * 2, I * 2 + 1);
    temp[I * 2] = idata[I * 2];
    temp[I * 2 + 1] = idata[I * 2 + 1];
    // printf("%d %f %f\n", I, temp[I * 2], temp[I * 2 + 1]);
    int offset = 1;

    for (int d = N / 2; d > 0; d >>= 1) {
        __syncthreads();  // sync within blocks
        if (I < d) {
            int ai = offset * (2 * I + 1) - 1;
            int bi = offset * (2 * I + 2) - 1;
            temp[bi] += temp[ai];
            // printf("%d %d %d %f %f\n", d, ai, bi, temp[ai], temp[bi]);
        }
        offset *= 2;
    }
    if (I == 0) temp[N - 1] = 0;
    // printf("%d %f %f\n", I, temp[I * 2], temp[I * 2 + 1]);
    
    for (int d = 1; d < N; d <<= 1) {
        offset /= 2;
        __syncthreads();
        if (I < d) {
            int ai = offset * (2 * I + 1) - 1;
            int bi = offset * (2 * I + 2) - 1;

            ValueType t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    odata[I * 2] = temp[I * 2];
    odata[I * 2 + 1] = temp[I * 2 + 1];
    // printf("%d %f %f\n", I, odata[I * 2], odata[I * 2 + 1]);
}

// N = 2^k and N < 2048
void pfxsum_v2_allshare(int N, ValueType* vals, ValueType* pfx) {
    dim3 block((N + 1) / 2);
    dim3 grid(1);

    rs_allshare<<<grid, block, N * sizeof(ValueType)>>>(N, vals, pfx);
}