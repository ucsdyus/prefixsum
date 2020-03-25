#ifndef HELPER_H_
#define HELPER_H_

#define ValueType float


struct CmdOptions {
    // -rd
    int use_rand;
    // -v valuie
    // 0: host (CPU)
    // 1: naive
    // 2: fast
    int version;
    // -n value
    int n;
    // -r value
    int reps;

    ValueType eps;
};

void genData(int N, ValueType* vals, int use_rand);

void cmdline(int argc, char *argv[], CmdOptions& cmd_opt);

double getTime();

double getGflops(int flops_per_rep, int reps, double runtime);

void perfString(double runtime, double gflops, const CmdOptions& cmd_opt);

void checkCUDAError(const char *msg);

int ReportDevice();

#endif  // HELPER_H_