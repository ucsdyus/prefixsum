#ifdef LINUX

#include <sys/time.h>
#include <getopt.h>

#else

#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64 
#include "getopt.h"

int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
    // until 00:00:00 January 1, 1970 
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime( &system_time );
    SystemTimeToFileTime( &system_time, &file_time );
    time =  ((uint64_t)file_time.dwLowDateTime )      ;
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
    return 0;
}
#endif  // endif LINUX

#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <cuda.h>
#include "helper.h"


void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }     
}

int ReportDevice() {
    int number_of_devices;
    cudaError_t  errCode = cudaGetDeviceCount(&number_of_devices);
	if ((errCode ==  cudaErrorNoDevice) || (errCode == cudaErrorInsufficientDriver)) {
	   printf("\n *** There are no available devices.\n");
	   printf("     Either you are not attached to a compute node or\n");
	   printf("     are not running in an appropraite batch queue.\n");
	   printf("\n Exiting...\n\n");
	   exit(EXIT_FAILURE);
	}
	printf("# devices: %d\n",number_of_devices);
    if (number_of_devices > 1) {
        printf("\n%d Devices\n",number_of_devices);
        int device_number;
        for (device_number = 0; device_number < number_of_devices; device_number++) {
            cudaDeviceProp deviceProp;
            assert(cudaSuccess == cudaGetDeviceProperties(&deviceProp, device_number));
            printf("Device # %d: capability %d.%d, %d cores\n",
                device_number, deviceProp.name, deviceProp.major,
                deviceProp.minor, deviceProp.multiProcessorCount);
        }
        printf("\n");
    }
    // get number of SMs on this GPU
    int devID;
    cudaGetDevice(&devID);
    cudaDeviceProp deviceProp;
    assert(cudaSuccess == cudaGetDeviceProperties(&deviceProp, devID));
    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
        printf("There is no device supporting CUDA.\n");
        cudaThreadExit();
    }
	printf("\nDevice is a %s, capability: %d.%d\n",  deviceProp.name, deviceProp.major, deviceProp.minor);

	printf("Clock speed: %f MHz\n",((double)deviceProp.clockRate)/1000);
    printf("# cores: %d\n",  deviceProp.multiProcessorCount);
	double gb = 1024*1024*1024;
    printf("\nGlobal memory: %fGB\n", ((double)deviceProp.totalGlobalMem)/gb);
	printf("Memory Clock Rate (MHz): %f\n", (double)deviceProp.memoryClockRate/1000);
	printf("Memory Bus Width (bits): %d\n", deviceProp.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s): %f\n", 2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1.0e6);
    printf("L2 Cache size: (KB): %f\n", (double)deviceProp.l2CacheSize/1024);
    if (deviceProp.ECCEnabled) {
	    printf("ECC Enabled\n");
    } else {
        printf("ECC NOT Enabled\n");
    }

    if (deviceProp.asyncEngineCount == 1)
       printf("Device can concurrently copy memory between host and device while executing a kernel\n");
    else if (deviceProp.asyncEngineCount == 2)
	    printf("Device can concurrently copy memory between host and device in both directions\n     and execute a kernel at the same time\n");
    else if (deviceProp.asyncEngineCount == 0){
	   printf("Device CANNOT copy memory between host and device while executing a kernel.\n");
	   printf("Device CANNOT copy memory between host and device in both directions at the same time.\n");
	}
    if (deviceProp.unifiedAddressing == 1) {
	  printf("Device shares a unified address space with the host\n");
    } else {
      printf("Device DOES NOT share a unified address space with the host\n");
    }

	cudaSharedMemConfig sMemConfig;
	assert(cudaSuccess == cudaDeviceGetSharedMemConfig(&sMemConfig));
	printf("Device Shared Memory Config (override with -D or -S) = %s\n",
	       (sMemConfig == cudaSharedMemBankSizeDefault) ? "Shared Mem Bank Size Default" :
	       (sMemConfig == cudaSharedMemBankSizeFourByte) ? "Shared Mem Bank Size 4B" :
	       (sMemConfig == cudaSharedMemBankSizeEightByte) ? "Shared Mem Bank Size 4B" :
	       "Unknown value returned from cudaDeviceGetSharedMemConfig");

	printf("\n --------- \n");

    int driverVersion, runtimeVersion;
	assert(cudaSuccess == cudaDriverGetVersion(&driverVersion));
	assert(cudaSuccess == cudaRuntimeGetVersion(&runtimeVersion));
    printf("CUDA Driver version: %d, runtime version: %d\n\n", driverVersion, runtimeVersion);

	return(100 * deviceProp.major + deviceProp.minor);
}

double getTime() {
    static const double kMicro = 1.0e-6;

    struct timeval TV;

	const int RC = gettimeofday(&TV, NULL);
	if(RC == -1) {
		printf("ERROR: Bad call to gettimeofday\n");
		return(-1);
	}

	return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );
}

double getGflops(int flops_per_rep, int reps, double runtime) {
        long long flops =  (long long) flops_per_rep * (long long) reps;
        double flop_rate = (double) flops / runtime;
        return flop_rate / 1.0e9;
}

void perfString(double runtime, double gflops, const CmdOptions& cmd_opt) {
    char use_rand = cmd_opt.use_rand ? 'Y' : 'N';
    
    printf("\n          N      Reps        T         GF     Vers    Rnd?\n");
    printf("@     %5d     %4d     %6.1f    %6.1f      %1d       %1c\n\n",
        cmd_opt.n, cmd_opt.reps, runtime, gflops, cmd_opt.version, use_rand);
}

void cmdline(int argc, char *argv[], CmdOptions& cmd_opt) {
    cmd_opt.use_rand = 0;
    cmd_opt.version = 0;
    cmd_opt.n = 8;
    cmd_opt.reps = 10;
    cmd_opt.eps = (ValueType) 1e-6;


    static struct option long_options[] = {
        {"n", required_argument, 0, 'n'},
        {"r", required_argument, 0, 'r'},
        {"v", required_argument, 0, 'v'},
        {"e", required_argument, 0, 'e'},
        {"d", no_argument, 0, 'd'},
    };

    
    for(int ac=1;ac<argc;ac++) {
        int c;
        while ((c = getopt_long(argc,argv,"n:r:v:e:d", long_options, NULL)) != -1){
            switch (c) {
                // Problem size
                case 'n':
                    cmd_opt.n = atoi(optarg);
                    break;

                // Number of repititions
                case 'r':
                    cmd_opt.reps = atoi(optarg);
                    break;
                
                // Version
                case 'v':
                    cmd_opt.version = atoi(optarg);
                    break;
                
                // Tolerance
                case 'e':
                    double tmp;
                    sscanf(optarg,"%lf",&tmp);
                    cmd_opt.eps = (ValueType) tmp;
                    break;
            
                // Use random
                case 'd':
                    cmd_opt.use_rand = 1;
                    break;

                // Error
                default:
                    printf("Usage: [-n <domain size>] [-r <reps>] [-v <version>] [-e <epsilon>] [-d <use rand>]\n");
                    exit(-1);
            }
        }
    }
}


void genData(int N, ValueType* vals, int use_rand) {
    for (int i = 0; i < N; ++i) {
        if (use_rand == 0) {
            vals[i] = 1;
        } else {
            ValueType t = (ValueType) rand() / (ValueType) (RAND_MAX);
            vals[i] = t * 2 - 1;
        }
    }
}