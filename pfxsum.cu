#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include "helper.h"

// L2
void verify(int N, ValueType* calc, ValueType* gt, ValueType eps);

// pfxsum kernels
void pfxsum_host(int N, ValueType* vals, ValueType* pfx);

void pfxsum_v1_allshare(int N, ValueType* vals, ValueType* pfx);

void pfxsum_v2_allshare(int N, ValueType* vals, ValueType* pfx);

void pfxsum_v3_thrust(int N, ValueType* vals, ValueType* pfx);

int main(int argc, char** argv) {
    CmdOptions cmd_opt;
    cmdline(argc, argv, cmd_opt);
    int capability = ReportDevice();
    printf("GPU capacity: %d\n", capability);

    ValueType* host_vals = new ValueType[cmd_opt.n];
    genData(cmd_opt.n, host_vals, cmd_opt.use_rand);

    ValueType* ground_truth = new ValueType[cmd_opt.n];
    pfxsum_host(cmd_opt.n, host_vals, ground_truth);

    ValueType* res = new ValueType[cmd_opt.n];

    ValueType* cuda_vals = NULL;
    ValueType* cuda_pfx = NULL;
    
    cudaMalloc(&cuda_vals, cmd_opt.n * sizeof(ValueType));
    checkCUDAError("Error allocating device memory for values");
    cudaMalloc(&cuda_pfx, cmd_opt.n * sizeof(ValueType));
    checkCUDAError("Error allocating device memory for prefix sum");

    cudaMemcpy(cuda_vals, host_vals, cmd_opt.n * sizeof(ValueType), cudaMemcpyHostToDevice);
    checkCUDAError("Error copying values from host to device");
    cudaMemset(cuda_pfx, 0, cmd_opt.n * sizeof(ValueType));
    checkCUDAError("Error set prefix sum to 0");
    
    cudaThreadSynchronize();
    double t_device = -getTime();

    switch(cmd_opt.version) {
        case 0:
            for (int i = 0; i < cmd_opt.reps; ++i) {
                pfxsum_host(cmd_opt.n, host_vals, res);
            }
            break;
        case 1:
            for (int i = 0; i < cmd_opt.reps; ++i) {
                pfxsum_v1_allshare(cmd_opt.n, cuda_vals, cuda_pfx);
            }
            break;
        case 2:
            for (int i = 0; i < cmd_opt.reps; ++i) {
                pfxsum_v2_allshare(cmd_opt.n, cuda_vals, cuda_pfx);
            }
            break;
        case 3:
            for (int i = 0; i < cmd_opt.reps; ++i) {
                pfxsum_v3_thrust(cmd_opt.n, cuda_vals, cuda_pfx);
            }
            break;
        default:
            std::cout << "Not Implemented Error: version " << cmd_opt.version << std::endl;
            exit(-1);
    }

    cudaThreadSynchronize();
    t_device += getTime();

    checkCUDAError("Error in cuda kernel");
    switch(cmd_opt.version) {
        case 1: case 2: case 3:
        cudaMemcpy(res, cuda_pfx, cmd_opt.n * sizeof(ValueType), cudaMemcpyDeviceToHost);
        checkCUDAError("Error copying values from device to host");
        break;
    }
    cudaFree(cuda_vals);
    checkCUDAError("Error releasing values");
    cudaFree(cuda_pfx);
    checkCUDAError("Error releasing prefix sum");

    double gflops_d = getGflops(cmd_opt.n, cmd_opt.reps, t_device);
    printf("Computation time: %f sec. [%f gflops]\n", t_device, gflops_d);
    perfString(t_device, gflops_d, cmd_opt);

    verify(cmd_opt.n, res, ground_truth, cmd_opt.eps);
    delete [] host_vals;
    delete [] ground_truth;
    delete [] res;
    return 0;
}

void verify(int N, ValueType* res, ValueType* gt, ValueType eps) {
    ValueType err = 0;
    for (int i = 0; i < N; ++i) {
        err += (res[i] - gt[i]) * (res[i] - gt[i]);
    }
    err /= (ValueType) N;
    if (err > eps) {
        std::cout << "*** a total of error: " << err
            << " exceeds eps: " << eps << std::endl;
        
        std::cout << ">>> CALC: " << res[0];
        for (int i = 1; i < N; ++i) {
            std::cout << " " << res[i];
        }
        std::cout << std::endl;

        std::cout << ">>> GT: " << gt[0];
        for (int i = 1; i < N; ++i) {
            std::cout << " " << gt[i];
        }
        std::cout << std::endl;
    } else {
        std::cout << "*** MSE < eps: " << eps << std::endl;
        std::cout << "*** answer verified" << std::endl;
    }
}