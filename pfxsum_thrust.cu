#include <cstdio>
#include <cuda.h>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "helper.h"


void pfxsum_v3_thrust(int N, ValueType* vals, ValueType* pfx) {
    thrust::exclusive_scan(thrust::device, vals, vals + N, pfx);
}