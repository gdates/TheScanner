/****************************************************************************************
 * gpu5090_scan.cu – prefix-scanner pentru 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
 * ------------------------------------------------------------------------------------ *
 *  • CUDA 12+  (compilează: nvcc -O3 -std=c++17 -arch=sm_90 -o gpu5090_scan gpu5090_scan.cu)
 *  • Host iterează prefixurile 0x40 … 0x7F   (8 biţi high  |  56 biţi low în kernel)
 *  • Sequenţial vs random cu --mode  sequential|random
 *  • BATCH-32  (schimbă pentru tuning)
 * ------------------------------------------------------------------------------------ */

#include <cuda.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <chrono>

/*──────────────────────── SHA-256 constants ──────────────────────*/
__device__ __constant__ uint32_t K256[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
  0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
  0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
  0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
  0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
  0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
  0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
  0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
  0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

/*────────────────────  stubs rapide (înlocuieşte) ─────────────────*/
struct Point { uint32_t x[8], y[8]; };
__device__ __forceinline__ void scalar_mul(uint64_t, Point&) {}   // TODO: Win-4
__device__ __forceinline__ void sha256   (const uint8_t*,int,uint8_t*){} // TODO
__device__ __forceinline__ void ripemd160(const uint8_t*,int,uint8_t*){} // TODO
/*──────────────────────────────────────────────────────────────────*/

#define BATCH 32

/*──────────────────── Kernel: primeşte 8-bit prefix ───────────────*/
__global__ void scan56(uint8_t high, uint64_t range,
                       const uint32_t tgt[5], unsigned long long *hit,
                       bool rnd, uint64_t seed)
{
    uint64_t tid    = blockIdx.x*blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x * BATCH;

    curandStatePhilox4_32_10_t rs;
    if(rnd) curand_init(seed+tid, 0, 0, &rs);

    for(uint64_t base=tid*BATCH; base<range; base+=stride){
        #pragma unroll
        for(int j=0;j<BATCH;++j){
            uint64_t low56 = rnd? (curand(&rs) % range) : (base+j);
            uint64_t priv  = (uint64_t)high<<56 | low56;     // 64-bit cheia completă

            Point P; scalar_mul(priv, P);

            uint8_t comp[33]; comp[0]=0x02 | (P.y[0]&1);
            memcpy(comp+1, P.x, 32);

            uint8_t h256[32]; sha256(comp,33,h256);
            uint8_t h160[20]; ripemd160(h256,32,h160);
            const uint32_t* h32=reinterpret_cast<const uint32_t*>(h160);

            bool ok = h32[0]==tgt[0] && h32[1]==tgt[1] &&
                      h32[2]==tgt[2] && h32[3]==tgt[3] && h160[16]==((uint8_t*)tgt)[16];
            if(ok){ atomicExch(hit,(unsigned long long)priv); return; }
        }
    }
}

/*──────────────────── Helpers + main - host loop ─────────────────*/
static uint32_t swap32(uint32_t x){ return __builtin_bswap32(x);}
int main(int argc,char**argv)
{
    /* Target Hash160 pentru 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU */
    uint32_t tgt[5] = {
        swap32(0xf6f5431d), swap32(0x25bbf7b1),
        swap32(0x2e8add9a), swap32(0xf5e3475c),
        swap32(0xb8a5a044)
    };

    bool rnd=false; int blocks=4096,threads=256;
    for(int i=1;i<argc;++i)
        if(!strcmp(argv[i],"--mode")) rnd=!strcmp(argv[++i],"random");
        else if(!strcmp(argv[i],"--blocks"))  blocks=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--threads")) threads=atoi(argv[++i]);

    /* Prefix 0x40–0x7F => 64×2^56 chei */
    const uint64_t RANGE56 = 0x0100000000000000ULL; // 2^56
    unsigned long long *dHit; cudaMalloc(&dHit,8);

    for(uint8_t high=0x40; high<=0x7F; ++high){
        cudaMemset(dHit, 0xFF, 8);
        uint64_t seed=std::chrono::high_resolution_clock::now().time_since_epoch().count();

        cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);

        scan56<<<blocks,threads>>>(high,RANGE56,tgt,dHit,rnd,seed);

        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms,t0,t1);

        unsigned long long res;
        cudaMemcpy(&res,dHit,8,cudaMemcpyDeviceToHost);

        double kps=RANGE56/(ms/1000.0);
        printf("pref 0x%02x  %.2f keys/s  (%.1f ms)\n",high,kps,ms);

        if(res!=0xFFFFFFFFFFFFFFFFULL){
            printf("[HIT] high=%02x  key=%016llx\n",high,res);
            break;
        }
    }
    cudaFree(dHit);
    return 0;
}
