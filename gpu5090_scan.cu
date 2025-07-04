// gpu5090_scan.cu – clean compile baseline (stubs for ECC + hash)
// ---------------------------------------------------------------
//  • CUDA 12+, testat pe sm_89 / sm_90
//  • Toate constantele SHA în __constant__ la file-scope
//  • atomicExch folosește unsigned long long* (64-bit)
//  • ECC, SHA-256, RIPEMD-160 lăsate ca TODO stubs – codul rulează și cronometrează
// ----------------------------------------------------------------

#include <cuda.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <chrono>

// ===== SHA-256 round constants (în memorie constantă) ===========
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

// ===== Stubs (înlocuiește cu implementările rapide) =============
struct Point { uint32_t x[8], y[8]; };
__device__ __forceinline__ void scalar_mul(uint64_t, Point&) {}               // TODO
__device__ __forceinline__ void sha256   (const uint8_t*, int, uint8_t*) {}    // TODO
__device__ __forceinline__ void ripemd160(const uint8_t*, int, uint8_t*) {}    // TODO
// ================================================================

#define BATCH 32

// ---------------- GPU kernel ------------------------------------
__global__ void scan(uint64_t start, uint64_t range, const uint32_t tgt[5],
                     unsigned long long *hit, bool rnd, uint64_t seed)
{
    uint64_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x * BATCH;

    curandStatePhilox4_32_10_t rs;
    if(rnd) curand_init(seed + tid, 0, 0, &rs);

    for(uint64_t base = tid * BATCH; base < range; base += stride){
        #pragma unroll
        for(int j = 0; j < BATCH; ++j){
            uint64_t k = rnd ? (curand(&rs) % range) : (base + j);
            k += start;

            Point P; scalar_mul(k, P);

            uint8_t comp[33]; comp[0] = 0x02 | (P.y[0] & 1);
            memcpy(comp + 1, P.x, 32);

            uint8_t h256[32]; sha256(comp, 33, h256);
            uint8_t h160[20]; ripemd160(h256, 32, h160);
            const uint32_t *h32 = reinterpret_cast<const uint32_t*>(h160);

            bool ok = h32[0] == tgt[0] && h32[1] == tgt[1] &&
                      h32[2] == tgt[2] && h32[3] == tgt[3] &&
                      h160[16] == ((const uint8_t*)tgt)[16];   // simplu demo

            if(ok){
                atomicExch(hit, static_cast<unsigned long long>(k));
                return;
            }
        }
    }
}

// ---------------- host helpers ----------------------------------
static uint64_t hex2u64(const char* s){ return strtoull(s, nullptr, 16); }

int main(int argc, char** argv)
{
    uint64_t s = 0, e = 0;  int blocks = 4096, threads = 256; bool rnd = false;

    for(int i = 1; i < argc; ++i){
        if(!strcmp(argv[i], "--start"))    s = hex2u64(argv[++i]);
        else if(!strcmp(argv[i], "--end")) e = hex2u64(argv[++i]);
        else if(!strcmp(argv[i], "--blocks"))  blocks  = atoi(argv[++i]);
        else if(!strcmp(argv[i], "--threads")) threads = atoi(argv[++i]);
        else if(!strcmp(argv[i], "--mode"))    rnd = !strcmp(argv[++i], "random");
    }
    if(e <= s){ fprintf(stderr, "Bad range\n"); return 1; }

    // hash160 țintă = 0 (demo). Înlocuiește cu decodarea adresei Base58.
    uint32_t tgt[5] = {
    0x1d43f5f6,
    0xb1f7bb25,
    0x9add8a2e,
    0x5c47e3f5,
    0xb8a5a044
};

    unsigned long long *dHit;
    cudaMalloc(&dHit, 8);
    cudaMemset(dHit, 0xFF, 8);

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    scan<<<blocks, threads>>>(s, e - s + 1, tgt, dHit, rnd, seed);

    cudaEventRecord(t1); cudaEventSynchronize(t1);

    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    unsigned long long res; cudaMemcpy(&res, dHit, 8, cudaMemcpyDeviceToHost);

    double kps = (double)(e - s + 1) / (ms / 1000.0);
    printf("Scanned %.0f keys in %.2f ms  =>  %.2f keys/s\n",
           (double)(e - s + 1), ms, kps);

    if(res != 0xFFFFFFFFFFFFFFFFULL)
        printf("[HIT] %016llx\n", res);

    cudaFree(dHit);
    return 0;
}
