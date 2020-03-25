#include "helper.h"

void pfxsum_host(int N, ValueType* vals, ValueType* pfx) {
    pfx[0] = 0;
    for (int i = 1; i < N; ++i) {
        pfx[i] = pfx[i - 1] + vals[i - 1];
    }
}