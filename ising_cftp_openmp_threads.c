// ising_cftp_seeds_perbeta_limit.c
// Propp-Wilson (CFTP) for Ising (Windows-friendly, OpenMP threads).
// - Store-by-seed (saves 2 uint64 per sweep).
// - Circular buffer for sweeps (prepend cheaply).
// - Per-attempt timeout (seconds) and retries (attempts).
// - Per-BETA time budget (900 seconds = 15 minutes) — enforced per K and beta.
// - If an attempt fails/timeouts, the program continues; results are recorded for all tasks.
// Compile (MinGW / MSYS2):
// gcc -O3 -fopenmp -march=native -o ising_cftp_perbeta.exe ising_cftp_seeds_perbeta_limit.c -lm
//
// Run:
// ./ising_cftp_perbeta.exe
//
// Output: ising_cftp_seeds_perbeta_samples.csv

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/////////////////////// Parameters ///////////////////////
const int Ks[] = {10, 15, 20};
const int num_K = 3;
const double betas[] = {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
const int num_betas = 11;
const int samples_per_beta = 100;

// CFTP control
const int max_doublings = 40;         // safety cap
const int max_attempts = 3;           // attempts per sample
const double default_attempt_timeout_s = 3600.0; // seconds per attempt (10 min)

// Per-beta budget (15 minutes)
const double beta_time_budget_s = 15.0 * 60.0; // 900 seconds

// Parallelism control
int threads_normal = 12;   // threads for non-problematic betas
int threads_reduced = 2;   // threads for problematic betas (0.4 - 0.6)

// Output
const char *output_csv = "ising_cftp_seeds_perbeta_samples.csv";

// Microbenchmark toggle (set to 1 to run quick bench and exit)
const int RUN_MICROBENCH = 0;
const int MICROBENCH_SWEEPS = 10000;

//////////////////////////////////////////////////////////

// ---------------- RNG (xorshift128+) ------------------
typedef struct { uint64_t s[2]; } rng_state;
static inline uint64_t rng_next(rng_state *st) {
    uint64_t s1 = st->s[0];
    uint64_t s0 = st->s[1];
    uint64_t result = s0 + s1;
    st->s[0] = s0;
    s1 ^= s1 << 23;
    st->s[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
    return result + st->s[1];
}
static inline double rng_uniform01(rng_state *st) {
    uint64_t r = rng_next(st);
    uint64_t mant = r >> 11;
    const double inv = 1.0 / (double)(1ULL<<53);
    return (double)mant * inv;
}
static inline void rng_seed_from(uint64_t seed1, uint64_t seed2, rng_state *st) {
    st->s[0] = seed1 ? seed1 : 0x243F6A8885A308D3ULL;
    st->s[1] = seed2 ? seed2 : 0x13198A2E03707344ULL;
    for (int i=0;i<10;i++) rng_next(st);
}

//////////////////////////////////////////////////////////
// Sweep (seed-based) and circular buffer types
//////////////////////////////////////////////////////////
typedef struct {
    uint64_t seed_even;
    uint64_t seed_odd;
} sweep_seed_t;

typedef struct {
    sweep_seed_t *arr;   // capacity array
    size_t cap;          // capacity
    size_t head;         // index of oldest element within arr
    size_t count;        // number of stored sweeps
} sweepbuf_t;

// init sweep buffer
void sweepbuf_init(sweepbuf_t *b, size_t initial_cap) {
    b->cap = (initial_cap > 0) ? initial_cap : 16;
    b->arr = (sweep_seed_t*)malloc(sizeof(sweep_seed_t) * b->cap);
    b->head = 0;
    b->count = 0;
}

// free sweep buffer
void sweepbuf_free(sweepbuf_t *b) {
    if (b->arr) free(b->arr);
    b->arr = NULL; b->cap = 0; b->head = 0; b->count = 0;
}

// ensure capacity >= mincap
int sweepbuf_ensure_cap(sweepbuf_t *b, size_t mincap) {
    if (b->cap >= mincap) return 0;
    size_t newcap = b->cap;
    while (newcap < mincap) newcap *= 2;
    sweep_seed_t *newarr = (sweep_seed_t*)malloc(sizeof(sweep_seed_t)*newcap);
    if (!newarr) return -1;
    // copy existing elements starting at index 0 in order oldest->newest
    for (size_t i=0;i<b->count;i++) {
        size_t idx = (b->head + i) % b->cap;
        newarr[i] = b->arr[idx];
    }
    free(b->arr);
    b->arr = newarr;
    b->cap = newcap;
    b->head = 0;
    return 0;
}

// prepend one sweep_seed (as oldest element)
int sweepbuf_prepend(sweepbuf_t *b, sweep_seed_t sw) {
    if (b->count + 1 > b->cap) {
        if (sweepbuf_ensure_cap(b, b->cap * 2) != 0) return -1;
    }
    // new head index = (head - 1 + cap) % cap
    b->head = (b->head + b->cap - 1) % b->cap;
    b->arr[b->head] = sw;
    b->count++;
    return 0;
}

// iterate i=0..count-1: element at (head+i)%cap
static inline sweep_seed_t sweepbuf_get(const sweepbuf_t *b, size_t i) {
    return b->arr[(b->head + i) % b->cap];
}

//////////////////////////////////////////////////////////
// Helper index
static inline int idx2(int i, int j, int K) { return i*K + j; }

//////////////////////////////////////////////////////////
// Generate a sweep as two seeds (using rng_next)
void generate_sweep_seed(rng_state *rng, sweep_seed_t *out) {
    out->seed_even = rng_next(rng);
    out->seed_odd  = rng_next(rng);
}

//////////////////////////////////////////////////////////
// Apply one half-update using seed: regenerate uniforms on the fly
// This is serial (no inner OpenMP) to avoid nested threads.
void half_update_from_seed(const int8_t *config, int8_t *out_config, uint64_t seed,
                           const unsigned char *mask, int K, double beta) {
    int N = K*K;
    rng_state local;
    uint64_t seed2 = seed ^ 0x9e3779b97f4a7c15ULL;
    rng_seed_from(seed, seed2, &local);
    for (int idx = 0; idx < N; ++idx) {
        if (!mask[idx]) { out_config[idx] = config[idx]; continue; }
        double u = rng_uniform01(&local);
        int i = idx / K;
        int j = idx % K;
        int up_i = (i - 1 + K) % K;
        int down_i = (i + 1) % K;
        int left_j = (j - 1 + K) % K;
        int right_j = (j + 1) % K;
        int s = 0;
        s += config[idx2(up_i, j, K)];
        s += config[idx2(down_i, j, K)];
        s += config[idx2(i, left_j, K)];
        s += config[idx2(i, right_j, K)];
        double p_plus;
        if (beta == 0.0) p_plus = 0.5;
        else {
            double arg = 2.0 * beta * (double)s;
            p_plus = 1.0 / (1.0 + exp(-arg));
        }
        out_config[idx] = (u < p_plus) ? (int8_t)1 : (int8_t)-1;
    }
}

// Apply sweep given sweep_seed_t (even then odd)
void apply_sweep_from_seed(int8_t *config, int8_t *buffer, const sweep_seed_t *sw,
                           const unsigned char *even_mask, const unsigned char *odd_mask,
                           int K, double beta) {
    // even half
    half_update_from_seed(config, buffer, sw->seed_even, even_mask, K, beta);
    memcpy(config, buffer, K*K * sizeof(int8_t));
    // odd half
    half_update_from_seed(config, buffer, sw->seed_odd, odd_mask, K, beta);
    memcpy(config, buffer, K*K * sizeof(int8_t));
}

//////////////////////////////////////////////////////////
// CFTP core with seeds + circular buffer + timeout check
// returns:
//   0 -> success (sample_out filled, T_out set)
//   1 -> failed to coalesce (exceeded max_doublings)
//   2 -> timeout (elapsed > attempt_timeout_s)
//   3 -> error (allocation or other)
int run_cftp_one_sample_with_timeout(int K, double beta, int8_t *sample_out, int *T_out,
                                     double attempt_timeout_s, unsigned long base_seed) {
    int N = K*K;

    // allocate configs/buffer
    int8_t *top = (int8_t*)malloc(N);
    int8_t *bottom = (int8_t*)malloc(N);
    int8_t *buffer = (int8_t*)malloc(N);
    if (!top || !bottom || !buffer) {
        fprintf(stderr,"[ERROR] alloc configs failed\n");
        if (top) free(top); if (bottom) free(bottom); if (buffer) free(buffer);
        return 3;
    }
    for (int i=0;i<N;i++) { top[i] = 1; bottom[i] = -1; buffer[i] = 0; }

    // masks even/odd
    unsigned char *even_mask = (unsigned char*)malloc(N);
    unsigned char *odd_mask  = (unsigned char*)malloc(N);
    if (!even_mask || !odd_mask) {
        fprintf(stderr,"[ERROR] alloc masks failed\n");
        free(top); free(bottom); free(buffer);
        if (even_mask) free(even_mask); if (odd_mask) free(odd_mask);
        return 3;
    }
    for (int idx=0; idx<N; ++idx) {
        int i = idx / K;
        int j = idx % K;
        unsigned char m = ((i + j) % 2 == 0) ? 1 : 0;
        even_mask[idx] = m;
        odd_mask[idx]  = 1 - m;
    }

    // sweep buffer (using seeds only)
    sweepbuf_t sb;
    sweepbuf_init(&sb, 16);

    // RNG for generating seeds
    rng_state rng;
    uint64_t seed1 = ((uint64_t)time(NULL) ^ (uint64_t)base_seed) + (uint64_t)K*1315423911ULL;
    uint64_t seed2 = seed1 ^ 0x9e3779b97f4a7c15ULL;
    rng_seed_from(seed1, seed2, &rng);

    // initial T=1 sweep generated and prepended (oldest)
    size_t T = 1;
    for (size_t t=0;t<T;++t) {
        sweep_seed_t sw;
        generate_sweep_seed(&rng, &sw);
        if (sweepbuf_prepend(&sb, sw) != 0) {
            sweepbuf_free(&sb);
            free(top); free(bottom); free(buffer); free(even_mask); free(odd_mask);
            return 3;
        }
    }

    int doubling_count = 0;
    double t_start = omp_get_wtime();

    while (1) {
        // check timeout occasionally
        double now = omp_get_wtime();
        if (now - t_start > attempt_timeout_s) {
            // cleanup
            sweepbuf_free(&sb);
            free(top); free(bottom); free(buffer);
            free(even_mask); free(odd_mask);
            return 2; // timeout
        }

        // simulate from oldest..newest: i.e., sb.arr[head .. head+count-1]
        int8_t *top_c = (int8_t*)malloc(N);
        int8_t *bottom_c = (int8_t*)malloc(N);
        if (!top_c || !bottom_c) {
            fprintf(stderr,"[ERROR] alloc temp configs failed\n");
            if (top_c) free(top_c); if (bottom_c) free(bottom_c);
            sweepbuf_free(&sb);
            free(top); free(bottom); free(buffer); free(even_mask); free(odd_mask);
            return 3;
        }
        memcpy(top_c, top, N);
        memcpy(bottom_c, bottom, N);

        // apply sweeps in order
        for (size_t si = 0; si < sb.count; ++si) {
            sweep_seed_t sw = sweepbuf_get(&sb, si);
            apply_sweep_from_seed(top_c, buffer, &sw, even_mask, odd_mask, K, beta);
            apply_sweep_from_seed(bottom_c, buffer, &sw, even_mask, odd_mask, K, beta);
        }

        // check coalescence
        if (memcmp(top_c, bottom_c, N*sizeof(int8_t)) == 0) {
            // success
            memcpy(sample_out, top_c, N*sizeof(int8_t));
            *T_out = (int)T;
            free(top_c); free(bottom_c);
            sweepbuf_free(&sb);
            free(top); free(bottom); free(buffer);
            free(even_mask); free(odd_mask);
            return 0;
        }
        free(top_c); free(bottom_c);

        // not coalesced -> double
        doubling_count++;
        if (doubling_count % 5 == 0) {
            int tid = omp_get_thread_num();
            // Note: no sidx/attempt here (they are handled by caller printing), this is internal diagnostic.
            printf("[CFTP] thread=%d K=%d beta=%.3f doubling=%d T=%zu\n", tid, K, beta, doubling_count, T);
            fflush(stdout);
        }
        if (doubling_count > max_doublings) {
            sweepbuf_free(&sb);
            free(top); free(bottom); free(buffer);
            free(even_mask); free(odd_mask);
            return 1; // failed to coalesce within doublings
        }

        // prepend T new sweeps (generate T seeds)
        for (size_t t=0;t<T;++t) {
            sweep_seed_t sw;
            generate_sweep_seed(&rng, &sw);
            if (sweepbuf_prepend(&sb, sw) != 0) {
                sweepbuf_free(&sb);
                free(top); free(bottom); free(buffer);
                free(even_mask); free(odd_mask);
                return 3;
            }
        }
        T *= 2;
    } // while
}

//////////////////////////////////////////////////////////
// micro-benchmark: measure time per sweep for given K
void run_microbenchmark(int K, int trials) {
    printf("Running microbenchmark: K=%d, trials=%d\n", K, trials);
    int N = K*K;
    int8_t *config = (int8_t*)malloc(N);
    int8_t *buffer = (int8_t*)malloc(N);
    for (int i=0;i<N;i++) config[i] = (i % 2) ? 1 : -1;

    rng_state rng;
    uint64_t seed1 = (uint64_t)time(NULL) ^ 0x12345678ABCDEFULL;
    rng_seed_from(seed1, seed1 ^ 0x9e3779b97f4a7c15ULL, &rng);
    sweep_seed_t sw;
    generate_sweep_seed(&rng, &sw);

    unsigned char *even_mask = (unsigned char*)malloc(N);
    unsigned char *odd_mask  = (unsigned char*)malloc(N);
    for (int idx=0; idx<N; ++idx) {
        int i = idx / K;
        int j = idx % K;
        unsigned char m = ((i + j) % 2 == 0) ? 1 : 0;
        even_mask[idx] = m; odd_mask[idx] = 1 - m;
    }

    double t0 = omp_get_wtime();
    for (int t=0;t<trials;t++) {
        apply_sweep_from_seed(config, buffer, &sw, even_mask, odd_mask, K, 0.5);
    }
    double t1 = omp_get_wtime();
    double total = t1 - t0;
    double t_sweep = total / trials;
    printf("microbenchmark: total=%.6fs, t_sweep=%.9fs\n", total, t_sweep);

    free(config); free(buffer); free(even_mask); free(odd_mask);
}

//////////////////////////////////////////////////////////
// helper to convert sample to string
void sample_to_string(const int8_t *sample, int N, char *buf, size_t bufsize) {
    size_t pos = 0;
    for (int i=0;i<N;i++) {
        int written = snprintf(buf+pos, (pos<bufsize)?(bufsize-pos):0, "%d", (int)sample[i]);
        if (written < 0) break;
        pos += written;
        if (i < N-1) {
            if (pos+1 < bufsize) { buf[pos++] = ' '; buf[pos] = '\0'; }
        }
        if (pos + 50 > bufsize) break;
    }
}

//////////////////////////////////////////////////////////
// main: process per K, per beta. Enforce per-beta time budget.
// For each beta: spawn an OpenMP parallel for over samples_per_beta,
// but first set omp_set_num_threads depending on whether beta is problematic.
int main(int argc, char **argv) {
    // set no buffering on stdout for clearer logs
    setvbuf(stdout, NULL, _IONBF, 0);

    int omp_env = omp_get_max_threads();
    if (threads_normal > omp_env) threads_normal = omp_env;
    if (threads_reduced > threads_normal) threads_reduced = threads_normal;

    printf("Starting CFTP (seed-storage + circular buffer). threads_normal=%d threads_reduced=%d\n", threads_normal, threads_reduced);
    printf("Per-beta time budget = %.0fs (%.1f min). Per-attempt timeout default = %.0fs, max_attempts=%d\n",
           beta_time_budget_s, beta_time_budget_s/60.0, default_attempt_timeout_s, max_attempts);

    if (RUN_MICROBENCH) {
        run_microbenchmark(10, MICROBENCH_SWEEPS);
        return 0;
    }

    FILE *csv = fopen(output_csv, "w");
    if (!csv) { fprintf(stderr,"Cannot open CSV file\n"); return 1; }
    fprintf(csv, "K,beta,sample_idx,attempt_final,coalescence_T,magnetization,spins,status,elapsed_s\n");
    fflush(csv);

    uint64_t base_seed_global = (uint64_t)time(NULL) ^ 0xCAFEBABEULL;

    long total_all = (long)num_K * (long)num_betas * (long)samples_per_beta;
    long completed_all = 0;

    for (int ik=0; ik < num_K; ++ik) {
        int K = Ks[ik];
        int N = K*K;

        for (int ib=0; ib < num_betas; ++ib) {
            double beta = betas[ib];
            // choose threads count for this beta
            int n_threads = (beta >= 0.4 && beta <= 0.6) ? threads_reduced : threads_normal;
            if (n_threads < 1) n_threads = 1;
            if (n_threads > omp_env) n_threads = omp_env;
            omp_set_num_threads(n_threads);

            printf("\n--- K=%d beta=%.3f: launching %d threads for %d samples (per-beta budget %.0fs) ---\n",
                   K, beta, n_threads, samples_per_beta, beta_time_budget_s);

            // per-beta start time and exhausted flag
            double beta_start = omp_get_wtime();
            volatile int beta_exhausted_flag = 0; // set to 1 when budget exhausted

            // parallel over sample indices for this specific beta
            #pragma omp parallel for schedule(dynamic,1)
            for (int sidx = 0; sidx < samples_per_beta; ++sidx) {
                int thread_id = omp_get_thread_num();

                // if budget already exhausted, record BETA_TIMEOUT for this sample
                double now_check = omp_get_wtime();
                if (now_check - beta_start >= beta_time_budget_s) {
                    // mark exhausted (one-time)
                    #pragma omp critical
                    {
                        beta_exhausted_flag = 1;
                    }
                    // record BETA_TIMEOUT
                    #pragma omp critical
                    {
                        fprintf(csv, "%d,%.3f,%d,%d,%d,%.6f,%s,%s,%.6f\n",
                                K, beta, sidx, 0, -1, NAN, "(none)", "BETA_TIMEOUT", 0.0);
                        fflush(csv);
                        completed_all++;
                        double pct = 100.0 * (double)completed_all / (double)total_all;
                        static double start_time = 0;
                        if (start_time == 0) start_time = omp_get_wtime();
                        double elapsed_tot = omp_get_wtime() - start_time;
                        double eta = (completed_all>0) ? (elapsed_tot / (double)completed_all) * ((double)total_all - completed_all) : 0.0;
                        printf("Progress: %ld/%ld (%.2f%%) - elapsed %.0fs - ETA %.0fs (thread=%d K=%d beta=%.3f sidx=%d) [BETA_TIMEOUT]\n",
                               completed_all, total_all, pct, elapsed_tot, eta, thread_id, K, beta, sidx);
                    }
                    continue;
                }

                int success = 0;
                int final_T = -1;
                int attempt;
                double elapsed_accum = 0.0;
                char status_buf[32] = "UNKNOWN";

                int8_t *sample = (int8_t*)malloc(N);
                if (!sample) {
                    // allocation failed: record error and continue
                    #pragma omp critical
                    {
                        fprintf(csv, "%d,%.3f,%d,%d,%d,%.6f,%s,%s,%.6f\n",
                                K, beta, sidx, 0, -1, NAN, "(none)", "ERROR_ALLOC", 0.0);
                        fflush(csv);
                        completed_all++;
                        double pct = 100.0 * (double)completed_all / (double)total_all;
                        static double start_time = 0;
                        if (start_time == 0) start_time = omp_get_wtime();
                        double elapsed_tot = omp_get_wtime() - start_time;
                        double eta = (completed_all>0) ? (elapsed_tot / (double)completed_all) * ((double)total_all - completed_all) : 0.0;
                        printf("Progress: %ld/%ld (%.2f%%) - elapsed %.0fs - ETA %.0fs (thread=%d K=%d beta=%.3f sidx=%d) [ERROR_ALLOC]\n",
                               completed_all, total_all, pct, elapsed_tot, eta, thread_id, K, beta, sidx);
                    }
                    continue;
                }

                for (attempt = 1; attempt <= max_attempts; ++attempt) {
                    // check whether beta budget remains before starting this attempt
                    double now = omp_get_wtime();
                    if (now - beta_start >= beta_time_budget_s) {
                        // budget exhausted
                        #pragma omp critical
                        {
                            beta_exhausted_flag = 1;
                        }
                        strcpy(status_buf, "BETA_TIMEOUT");
                        break;
                    }
                    // compute remaining time for beta
                    double rem = beta_time_budget_s - (now - beta_start);
                    // choose attempt timeout = min(default_attempt_timeout_s, rem)
                    double this_attempt_timeout = default_attempt_timeout_s;
                    if (rem < this_attempt_timeout) this_attempt_timeout = rem;
                    if (this_attempt_timeout < 0.5) {
                        // not enough time left to attempt
                        #pragma omp critical
                        {
                            beta_exhausted_flag = 1;
                        }
                        strcpy(status_buf, "BETA_TIMEOUT");
                        break;
                    }

                    // run attempt with this_attempt_timeout
                    double t0 = omp_get_wtime();
                    int rc = run_cftp_one_sample_with_timeout(K, beta, sample, &final_T, this_attempt_timeout,
                                                              (unsigned long)(base_seed_global + (uint64_t)ik*100000 + (uint64_t)ib*1000 + (uint64_t)sidx*10 + (uint64_t)attempt));
                    double t1 = omp_get_wtime();
                    double elapsed = t1 - t0;
                    elapsed_accum += elapsed;

                    if (rc == 0) {
                        // success
                        success = 1;
                        strcpy(status_buf, "OK");
                        break;
                    } else if (rc == 2) {
                        // attempt timed out (but there may still be budget left)
                        strcpy(status_buf, "TIMEOUT");
                        // continue to next attempt if attempts remain and budget allows
                    } else if (rc == 1) {
                        // failed to coalesce (max_doublings)
                        strcpy(status_buf, "NO_COALESCE");
                        break;
                    } else if (rc == 3) {
                        strcpy(status_buf, "ERROR");
                        break;
                    } else {
                        strcpy(status_buf, "ERROR");
                        break;
                    }

                    // after attempt check if beta budget exhausted
                    now = omp_get_wtime();
                    if (now - beta_start >= beta_time_budget_s) {
                        #pragma omp critical
                        {
                            beta_exhausted_flag = 1;
                        }
                        strcpy(status_buf, "BETA_TIMEOUT");
                        break;
                    }
                } // attempts loop

                // If after attempts we have beta_exhausted_flag set, mark remaining samples appropriately.
                if (!success && strcmp(status_buf, "BETA_TIMEOUT") == 0) {
                    // nothing else, status already set
                }

                // prepare CSV entry
                #pragma omp critical
                {
                    if (success) {
                        double mag = 0.0;
                        for (int i=0;i<N;i++) mag += (double)sample[i];
                        mag /= (double)N;
                        size_t bufsize = N * 3 + 128;
                        char *buf = (char*)malloc(bufsize);
                        if (buf) {
                            sample_to_string(sample, N, buf, bufsize);
                            fprintf(csv, "%d,%.3f,%d,%d,%d,%.6f,\"%s\",%s,%.6f\n",
                                    K, beta, sidx, attempt, final_T, mag, buf, status_buf, elapsed_accum);
                            free(buf);
                        } else {
                            fprintf(csv, "%d,%.3f,%d,%d,%d,%.6f,%s,%s,%.6f\n",
                                    K, beta, sidx, attempt, final_T, mag, "(no-spins)", status_buf, elapsed_accum);
                        }
                    } else {
                        // failure/timeouts
                        fprintf(csv, "%d,%.3f,%d,%d,%d,%.6f,%s,%.6f\n",
                                K, beta, sidx, attempt-1, (final_T>0)?final_T:-1, NAN, status_buf, elapsed_accum);
                    }
                    fflush(csv);

                    completed_all++;
                    double pct = 100.0 * (double)completed_all / (double)total_all;
                    static double start_time = 0;
                    if (start_time == 0) start_time = omp_get_wtime();
                    double elapsed_tot = omp_get_wtime() - start_time;
                    double eta = (completed_all>0) ? (elapsed_tot / (double)completed_all) * ((double)total_all - completed_all) : 0.0;
                    printf("Progress: %ld/%ld (%.2f%%) - elapsed %.0fs - ETA %.0fs (thread=%d K=%d beta=%.3f sidx=%d) status=%s\n",
                           completed_all, total_all, pct, elapsed_tot, eta, thread_id, K, beta, sidx, status_buf);
                    fflush(stdout);
                } // critical

                free(sample);
            } // parallel for (samples for this beta)

            // after finishing this beta, ensure any remaining samples (if beta_exhausted_flag) are marked:
            // (we handled per-sample above by checking start time; so no extra work needed)

            // small delay to let OS breathe if desired
            // (none)
        } // ib
    } // ik

    fclose(csv);
    printf("All done. CSV written to %s\n", output_csv);
    return 0;
}
