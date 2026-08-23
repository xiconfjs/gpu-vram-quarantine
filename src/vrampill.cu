// vrampill - quarantine defective VRAM cells on a consumer NVIDIA GPU so the
// rest of the card stays usable.
//
// WHY THIS EXISTS
// ---------------
// A GDDR6/GDDR6X cell that cannot hold a bit makes an otherwise healthy card
// untrustworthy for anything that fills VRAM. There is no vendor remedy on
// GeForce: NVIDIA's dynamic page retirement and row remapping are explicitly
// unsupported on all GeForce products, and GDDR6X has no on-die array ECC (the
// link CRC covers transmission only), so a stuck cell returns a wrong value
// with a perfectly valid checksum. It is invisible to every reporting path on
// the card.
//
// This program locates the defective cells and holds the small allocations
// containing them for the life of the process. VRAM is a shared physical
// resource: once a process owns a page, the driver cannot hand it to any other
// process. Everything else on the card then becomes safe to use, without the
// cooperation of whatever workload you actually want to run.
//
// WHAT IT DOES NOT DO
// -------------------
// It does not repair anything. It does not detect faults it never provoked. A
// clean search is NOT proof the card is healthy - see --quiet-seconds and the
// exit codes below.
//
// Build: nvcc -O3 -arch=sm_86 -o vrampill vrampill.cu     (adjust -arch)
//
// Exit codes:
//   0  quarantine active (or the target GPU has no detectable fault to hold,
//      only when --allow-clean is given)
//   2  CUDA error
//   3  could not allocate enough VRAM to search meaningfully
//   4  no fault found. This is NOT a pass - it means nothing was quarantined.
//   5  found more failing chunks than --max-bad allows; refusing to continue.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <csignal>
#include <cuda_runtime.h>

#define CK(x) do {                                                            \
    cudaError_t _e = (x);                                                     \
    if (_e != cudaSuccess) {                                                  \
        printf("[CUDA ERROR] %s:%d %s -> %s\n", __FILE__, __LINE__, #x,       \
               cudaGetErrorString(_e));                                       \
        fflush(stdout);                                                       \
        return 2;                                                             \
    }                                                                         \
} while (0)

#define MAXCHUNK 16384

__device__ __forceinline__
unsigned int genval(unsigned long long gidx, unsigned int pat, int mode)
{
    if (mode == 1) return (unsigned int)(gidx * 2654435761ULL) ^ pat;
    if (mode == 2) return (gidx & 1ULL) ? ~pat : pat;
    return pat;
}

__global__ void kfill(unsigned int* p, unsigned long long n,
                      unsigned long long base, unsigned int pat, int mode)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = genval(base + i, pat, mode);
}

#define NO_CAND 0xFFFFFFFFFFFFFFFFULL

// Per-chunk state lives on the device so that a whole sweep costs one host copy
// rather than one per chunk. With thousands of chunks that is the difference
// between a usable search and an unusably slow one.
//
// The first error in a chunk nominates a candidate element index. Subsequent
// errors either land on that same index (corroboration) or somewhere else in the
// chunk (recorded separately). A chunk is only *confirmed* once the same element
// has failed --confirm-hits times, which is what distinguishes a genuinely stuck
// cell from a one-off transient. Credit to Olari-A for the requirement that
// discovery and confirmation must agree on the same offset and bit.
__global__ void kcheck(const unsigned int* p, unsigned long long n,
                       unsigned long long base, unsigned int pat, int mode,
                       int cidx, unsigned int* chunkhits,
                       unsigned long long* chunkidx, unsigned int* chunkxor,
                       unsigned int* samehits, unsigned int* mismatch)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        unsigned int want = genval(base + i, pat, mode);
        unsigned int got  = p[i];
        if (got != want) {
            atomicAdd(&chunkhits[cidx], 1u);
            unsigned long long prev =
                atomicCAS((unsigned long long*)&chunkidx[cidx], NO_CAND, i);
            if (prev == NO_CAND || prev == i) {
                atomicAdd(&samehits[cidx], 1u);
                atomicOr(&chunkxor[cidx], want ^ got);
            } else {
                atomicAdd(&mismatch[cidx], 1u);
            }
        }
    }
}

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char** argv)
{
    int    dev         = 0;
    size_t chunk_mib   = 8;
    double find_secs   = 1800.0;   // hard ceiling on the search
    double quiet_secs  = 180.0;    // stop once this long passes with nothing new
    int    max_bad     = 64;
    int    allow_clean = 0;
    int    confirm_hits = 2;   // repeats at the same offset before "confirmed"
    const char* ready_file = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) dev = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunk-mib") && i + 1 < argc) chunk_mib = (size_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--find-seconds") && i + 1 < argc) find_secs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--quiet-seconds") && i + 1 < argc) quiet_secs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--max-bad") && i + 1 < argc) max_bad = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--confirm-hits") && i + 1 < argc) confirm_hits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--allow-clean")) allow_clean = 1;
        else if (!strcmp(argv[i], "--ready-file") && i + 1 < argc) ready_file = argv[++i];
    }
    if (ready_file) unlink(ready_file);

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    CK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));

    size_t freeB = 0, totalB = 0;
    CK(cudaMemGetInfo(&freeB, &totalB));

    printf("=== vrampill ===\n");
    printf("device       : %d (%s)\n", dev, prop.name);
    printf("total VRAM   : %.2f GiB\n", totalB / 1073741824.0);
    printf("free VRAM    : %.2f GiB\n", freeB  / 1073741824.0);
    printf("chunk size   : %zu MiB\n", chunk_mib);
    printf("search       : up to %.0fs, stop after %.0fs with nothing new\n",
           find_secs, quiet_secs);
    fflush(stdout);

    static unsigned int*      chunk[MAXCHUNK];
    static unsigned long long celems[MAXCHUNK];
    static unsigned long long cbase[MAXCHUNK];
    static unsigned char      isbad[MAXCHUNK];   // confirmed: repeated at one offset
    static unsigned char      seen[MAXCHUNK];    // errored at least once, maybe transient
    int    nchunk = 0;
    size_t got_total = 0;

    size_t csize  = chunk_mib * 1048576ULL;
    size_t target = (freeB > 640ULL * 1048576ULL) ? freeB - 640ULL * 1048576ULL : 0;

    unsigned long long base = 0;
    while (nchunk < MAXCHUNK && got_total + csize <= target) {
        void* p = NULL;
        if (cudaMalloc(&p, csize) != cudaSuccess) { cudaGetLastError(); break; }
        chunk[nchunk]  = (unsigned int*)p;
        celems[nchunk] = csize / 4ULL;
        cbase[nchunk]  = base;
        isbad[nchunk]  = 0;
        seen[nchunk]   = 0;
        base      += celems[nchunk];
        got_total += csize;
        nchunk++;
    }
    printf("allocated    : %.2f GiB in %d chunks\n", got_total / 1073741824.0, nchunk);

    // A search that covers only part of the card can only ever quarantine
    // faults inside the part it covered.
    double cover = (double)got_total / (double)totalB;
    printf("coverage     : %.1f%% of total VRAM\n", cover * 100.0);
    if (cover < 0.80) {
        printf("\nABORT: only %.1f%% of VRAM could be allocated. Free the card first;\n"
               "anything holding VRAM hides the region it occupies from this search.\n",
               cover * 100.0);
        return 3;
    }
    printf("\n");
    fflush(stdout);

    unsigned int       *d_hits, *d_xor, *d_same, *d_mism;
    unsigned long long *d_idx;
    CK(cudaMalloc(&d_hits, sizeof(unsigned int) * nchunk));
    CK(cudaMalloc(&d_xor,  sizeof(unsigned int) * nchunk));
    CK(cudaMalloc(&d_same, sizeof(unsigned int) * nchunk));
    CK(cudaMalloc(&d_mism, sizeof(unsigned int) * nchunk));
    CK(cudaMalloc(&d_idx,  sizeof(unsigned long long) * nchunk));
    CK(cudaMemset(d_hits, 0, sizeof(unsigned int) * nchunk));
    CK(cudaMemset(d_xor,  0, sizeof(unsigned int) * nchunk));
    CK(cudaMemset(d_same, 0, sizeof(unsigned int) * nchunk));
    CK(cudaMemset(d_mism, 0, sizeof(unsigned int) * nchunk));
    // 0xFF... = NO_CAND, i.e. no candidate offset nominated yet.
    CK(cudaMemset(d_idx, 0xFF, sizeof(unsigned long long) * nchunk));

    unsigned int* h_hits = (unsigned int*)calloc(nchunk, sizeof(unsigned int));
    unsigned int* h_xor  = (unsigned int*)calloc(nchunk, sizeof(unsigned int));
    unsigned int* h_same = (unsigned int*)calloc(nchunk, sizeof(unsigned int));
    unsigned int* h_mism = (unsigned int*)calloc(nchunk, sizeof(unsigned int));
    unsigned long long* h_idx =
        (unsigned long long*)calloc(nchunk, sizeof(unsigned long long));
    if (!h_hits || !h_xor || !h_same || !h_mism || !h_idx) {
        printf("ABORT: host alloc failed\n"); return 2;
    }

    // Walking one and walking zero drive every bit position both high and low
    // at every address. A cell that only fails in one direction is invisible to
    // partial bit coverage.
    unsigned int pats[8 + 64];
    int npat = 0;
    const unsigned int base_pats[] = {
        0x00000000u, 0xFFFFFFFFu, 0xAAAAAAAAu, 0x55555555u,
        0xA5A5A5A5u, 0x5A5A5A5Au, 0xDEADBEEFu, 0x00FF00FFu
    };
    for (int i = 0; i < 8; i++) pats[npat++] = base_pats[i];
    for (int b = 0; b < 32; b++) pats[npat++] = (1u << b);
    for (int b = 0; b < 32; b++) pats[npat++] = ~(1u << b);

    int threads = 256;
    int blocks  = prop.multiProcessorCount * 16;

    printf("searching (faults are often temperature-gated; the card must warm up)...\n");
    fflush(stdout);

    double t0 = now_s();
    double last_new = t0;
    int    nbad = 0;
    unsigned int iter = 0;

    // Keep searching after the first hit. Stopping early was the single most
    // dangerous behaviour this tool could have: it would quarantine one cell,
    // report success, and leave any others live with no warning.
    while (!g_stop) {
        double el = now_s() - t0;
        if (el >= find_secs) break;
        if (nbad > 0 && (now_s() - last_new) >= quiet_secs) break;

        unsigned int pat  = pats[iter % npat];
        int          mode = (iter / npat) % 3;

        for (int c = 0; c < nchunk; c++)
            kfill<<<blocks, threads>>>(chunk[c], celems[c], cbase[c], pat, mode);
        CK(cudaDeviceSynchronize());

        for (int c = 0; c < nchunk; c++)
            kcheck<<<blocks, threads>>>(chunk[c], celems[c], cbase[c], pat, mode,
                                        c, d_hits, d_idx, d_xor, d_same, d_mism);
        CK(cudaDeviceSynchronize());

        CK(cudaMemcpy(h_hits, d_hits, sizeof(unsigned int) * nchunk, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_same, d_same, sizeof(unsigned int) * nchunk, cudaMemcpyDeviceToHost));
        for (int c = 0; c < nchunk; c++) {
            if (!h_hits[c]) continue;

            // Announce a chunk the first time it errors at all, but do not treat
            // it as confirmed until the same offset has failed repeatedly. One
            // hit could be a transient rather than a stuck cell.
            if (!seen[c]) {
                seen[c] = 1;
                CK(cudaMemcpy(h_idx, d_idx, sizeof(unsigned long long) * nchunk, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(h_xor, d_xor, sizeof(unsigned int) * nchunk, cudaMemcpyDeviceToHost));
                unsigned long long off = (cbase[c] + h_idx[c]) * 4ULL;
                printf("  [%.0fs] candidate: chunk %d, offset 0x%llX (%.2f GiB), xor 0x%08X"
                       " — needs %d hits to confirm\n",
                       now_s() - t0, c, off, off / 1073741824.0, h_xor[c], confirm_hits);
                fflush(stdout);
            }

            if (!isbad[c] && (int)h_same[c] >= confirm_hits) {
                isbad[c] = 1;
                nbad++;
                last_new = now_s();
                CK(cudaMemcpy(h_idx, d_idx, sizeof(unsigned long long) * nchunk, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(h_xor, d_xor, sizeof(unsigned int) * nchunk, cudaMemcpyDeviceToHost));
                unsigned long long off = (cbase[c] + h_idx[c]) * 4ULL;
                printf("  [%.0fs] CONFIRMED #%d: chunk %d, offset 0x%llX (%.2f GiB),"
                       " xor 0x%08X, %u hits at that offset\n",
                       now_s() - t0, nbad, c, off, off / 1073741824.0, h_xor[c], h_same[c]);
                fflush(stdout);
                if (nbad > max_bad) {
                    printf("\nABORT: more than --max-bad (%d) confirmed failing chunks.\n"
                           "This card is too damaged for quarantine to be sensible.\n", max_bad);
                    return 5;
                }
            }
        }
        iter++;
        if ((iter % 500) == 0) {
            printf("  [%.0fs] %u iters, %d confirmed\n", now_s() - t0, iter, nbad);
            fflush(stdout);
        }
    }

    double searched = now_s() - t0;

    CK(cudaMemcpy(h_idx,  d_idx,  sizeof(unsigned long long) * nchunk, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_xor,  d_xor,  sizeof(unsigned int) * nchunk, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_same, d_same, sizeof(unsigned int) * nchunk, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_mism, d_mism, sizeof(unsigned int) * nchunk, cudaMemcpyDeviceToHost));

    int nseen = 0, nunconf = 0;
    for (int c = 0; c < nchunk; c++) {
        if (!seen[c]) continue;
        nseen++;
        if (!isbad[c]) nunconf++;
    }

    if (nseen == 0) {
        printf("\nNo fault found in %.0fs / %u iterations.\n", searched, iter);
        printf("\nThis is NOT a clean bill of health. It means nothing was quarantined.\n"
               "A temperature-gated fault can need several minutes of load to appear,\n"
               "and a fault outside the allocated region cannot be seen at all.\n");
        if (!allow_clean) {
            printf("Exiting 4 so that callers fail closed. Pass --allow-clean to treat\n"
                   "this as success on a card you believe is healthy.\n");
            return 4;
        }
        printf("--allow-clean given; exiting 0 with no quarantine held.\n");
        if (ready_file) {
            FILE* f = fopen(ready_file, "w");
            if (f) { fprintf(f, "status=no-fault-found\ndevice=%d\n", dev); fclose(f); }
        }
        return 0;
    }

    // Quarantine anything that ever errored, confirmed or not. Holding an extra
    // 8 MiB costs nothing; releasing a chunk that turns out to be genuinely bad
    // costs silent corruption. Confirmation governs what gets *reported* as
    // established, not what gets held.
    size_t held = 0, freed = 0;
    for (int c = 0; c < nchunk; c++) {
        if (seen[c]) { held += celems[c] * 4ULL; continue; }
        cudaFree(chunk[c]);
        freed += celems[c] * 4ULL;
    }
    cudaFree(d_hits); cudaFree(d_xor); cudaFree(d_idx);
    cudaFree(d_same); cudaFree(d_mism);

    size_t f2 = 0, t2 = 0;
    cudaMemGetInfo(&f2, &t2);

    printf("\n=== QUARANTINE ACTIVE ===\n");
    printf("searched     : %.0fs, %u iterations\n", searched, iter);
    printf("confirmed    : %d chunk(s) (>= %d hits at one offset)\n", nbad, confirm_hits);
    if (nunconf)
        printf("unconfirmed  : %d chunk(s) errored but never repeated — held anyway\n", nunconf);
    printf("held         : %.1f MiB across %d chunk(s)\n", held / 1048576.0, nseen);
    printf("released     : %.2f GiB\n", freed / 1073741824.0);
    printf("free now     : %.2f GiB\n", f2 / 1073741824.0);

    printf("\n%-8s %-20s %-12s %-8s %-9s %s\n",
           "chunk", "offset", "xor", "hits", "elsewhere", "state");
    for (int c = 0; c < nchunk; c++) {
        if (!seen[c]) continue;
        printf("%-8d 0x%016llX  0x%08X   %-8u %-9u %s\n",
               c, (unsigned long long)((cbase[c] + h_idx[c]) * 4ULL), h_xor[c],
               h_same[c], h_mism[c], isbad[c] ? "CONFIRMED" : "unconfirmed");
    }
    if (nunconf)
        printf("\nUnconfirmed chunks errored once and never again at the same offset.\n"
               "That may be a transient rather than a stuck cell. They are quarantined\n"
               "regardless, but a longer run would settle it.\n");

    if (ready_file) {
        FILE* f = fopen(ready_file, "w");
        if (f) {
            fprintf(f, "status=active\ndevice=%d\nchunk_mib=%zu\n"
                       "confirmed_chunks=%d\nunconfirmed_chunks=%d\nheld_chunks=%d\n"
                       "held_mib=%.1f\nconfirm_hits=%d\n",
                    dev, chunk_mib, nbad, nunconf, nseen,
                    held / 1048576.0, confirm_hits);
            for (int c = 0; c < nchunk; c++)
                if (seen[c])
                    fprintf(f, "bad_offset=0x%llX xor=0x%08X hits=%u state=%s\n",
                            (unsigned long long)((cbase[c] + h_idx[c]) * 4ULL),
                            h_xor[c], h_same[c], isbad[c] ? "confirmed" : "unconfirmed");
            fclose(f);
            printf("ready marker : %s\n", ready_file);
        } else {
            printf("WARNING: could not write ready marker %s\n", ready_file);
        }
    }
    printf("\nHolding until killed. Verify the rest of the card separately.\n");
    fflush(stdout);

    while (!g_stop) sleep(2);

    printf("\nsignal received, releasing quarantine.\n");
    if (ready_file) unlink(ready_file);
    for (int c = 0; c < nchunk; c++) if (seen[c]) cudaFree(chunk[c]);
    return 0;
}
