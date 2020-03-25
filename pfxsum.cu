#include <iostream>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include "helper.h"

// L2
void verify(int N, ValueType* calc, ValueType* gt, ValueType eps);

// pfxsum kernels
void pfxsum_host(int N, ValueType* vals, ValueType* pfx);

int main(int argc, char** argv) {
    CmdOptions cmd_opt;
    cmdline(argc, argv, cmd_opt);
    int capability = ReportDevice();
    printf("GPU capacity: %d\n", capability);

    std::unique_ptr<ValueType[]> host_vals(new ValueType[cmd_opt.n]);
    genData(cmd_opt.n, host_vals.get(), cmd_opt.use_rand);

    std::unique_ptr<ValueType[]> ground_truth(new ValueType[cmd_opt.n]);
    pfxsum_host(cmd_opt.n, host_vals.get(), ground_truth.get());

    std::unique_ptr<ValueType[]> res(new ValueType[cmd_opt.n]);

    cudaThreadSynchronize();
    double t_device = -getTime();

    switch(cmd_opt.version) {
        case 0:
            for (int i = 0; i < cmd_opt.reps; ++i) {
                pfxsum_host(cmd_opt.n, host_vals.get(), res.get());
            }
            break;
        default:
            std::cout << "Not Implemented Error: version " << cmd_opt.version << std::endl;
            exit(-1);
    }

    cudaThreadSynchronize();
    t_device += getTime();

    double gflops_d = getGflops(cmd_opt.n, cmd_opt.reps, t_device);
    printf("Computation time: %f sec. [%f gflops]\n", t_device, gflops_d);
    perfString(t_device, gflops_d, cmd_opt);

    verify(cmd_opt.n, res.get(), ground_truth.get(), cmd_opt.eps);
    return 0;
}

void verify(int N, ValueType* res, ValueType* gt, ValueType eps) {
    ValueType err = 0;
    for (int i = 0; i < N; ++i) {
        err += (res[i] - gt[i]) * (res[i] - gt[i]);
    }
    err = std::sqrt(err);
    if (err > eps) {
        std::cout << "*** a total of error: " << err
            << " exceeds eps: " << eps << std::endl;
    } else {
        std::cout << "*** errro under eps: " << eps << std::endl;
        std::cout << "*** answer verified" << std::endl;
    }
}