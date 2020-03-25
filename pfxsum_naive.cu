#include <cuda.h>
#include "helper.h"

// put everything into shared memory.
__global__ void naive_scan_allshare(int N, ValueType *g_idata, ValueType *g_odata) {
    extern __shared__ ValueType temp[]; // allocated on invocation
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    int pout = 0, pin = 1;   // Load input into shared memory. 
    // This is exclusive scan, so shift right by one    
    // and set first element to 0   
    if (I < N) {
        if (I > 0) {
            temp[pout * N + I] = g_idata[I - 1];
        } else {
            temp[pout * N + I] = 0;
        }
    }
    
    __syncthreads();
    for (int offset = 1; offset < N; offset *= 2)   {
        if (I < N) {
            pout = 1 - pout; // swap double buffer indices
            pin = 1 - pout;
            if (I >= offset) {
                temp[pout * N + I] += temp[pin * N + I - offset];
            } else {
                temp[pout * N + I] = temp[pin * N + I];
            }
        }
        __syncthreads();   
    }
    if (I < N) {
        g_odata[I] = temp[pout * N + I]; // write output 
    }
} 

#define V1_MAX_THREADS 1024

void pfxsum_v1_allshare(int N, ValueType* vals, ValueType* pfx) {
    dim3 block;
    dim3 grid;
    if (N <= V1_MAX_THREADS) {
        block.x = N;
        grid.x = 1;
    } else {
        block.x = V1_MAX_THREADS;
        grid.x = (N + V1_MAX_THREADS - 1) / V1_MAX_THREADS;
    }

    naive_scan_allshare<<<grid, block, N * 2 * sizeof(ValueType)>>>(N, vals, pfx);
}