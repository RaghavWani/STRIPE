/*
 * rfi_mitigation_omp_fixed.c
 * C conversion of original C++ code by Raghav Wani (Last edited 18th Feb 2026)
 *
 * Compile:
 *   gcc -O2 -std=c11 -fopenmp -o rfi_mitigation_final rfi_mitigation_final.c -lm
 *
 * Control thread count at runtime:
 *   ./rfi_mitigation_final  <file> <block_size> <threshold> <num_threads>
 *   OMP_NUM_THREADS=8 ./rfi_mitigation_final  <file> <block_size> <threshold>
 *
 * ═══════════════════════════════════════════════════════════════════════
 * BUGS FIXED vs rfi_mitigation_omp.c
 * ═══════════════════════════════════════════════════════════════════════
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <omp.h>

/* =========================================================================
 * Portability helpers
 * ====================================================================== */
static float clampf(float v, float lo, float hi)
{
    if (v < lo)
        return lo;
    if (v > hi)
        return hi;
    return v;
}
static double now_usec(void) { return omp_get_wtime() * 1e6; }
static double now_msec(void) { return omp_get_wtime() * 1e3; }

/* =========================================================================
 * Error macros
 * ====================================================================== */
#define FATAL(msg)                           \
    do                                       \
    {                                        \
        fprintf(stderr, "Error: %s\n", msg); \
        exit(1);                             \
    } while (0)
#define FATALF(fmt, ...)                                  \
    do                                                    \
    {                                                     \
        fprintf(stderr, "Error: " fmt "\n", __VA_ARGS__); \
        exit(1);                                          \
    } while (0)

typedef struct
{
    double sum;
    double c;
} KahanSum;

static inline void kahan_add(KahanSum *ks, double v)
{
    double y = v - ks->c;
    double t = ks->sum + y;
    ks->c = (t - ks->sum) - y;
    ks->sum = t;
}

/* =========================================================================
 * Thread-local Box-Muller RNG  (Xorshift64 + Box-Muller)
 * ====================================================================== */
typedef struct
{
    unsigned long long state;
} Rng;

static void rng_init(Rng *r, int extra_seed)
{
    r->state = (unsigned long long)time(NULL) ^ (unsigned long long)(uintptr_t)r ^ ((unsigned long long)extra_seed * 6364136223846793005ULL);
    r->state ^= r->state << 13;
    r->state ^= r->state >> 7;
    r->state ^= r->state << 17;
}
static double rng_uniform(Rng *r)
{
    r->state ^= r->state << 13;
    r->state ^= r->state >> 7;
    r->state ^= r->state << 17;
    return (double)(r->state >> 11) / (double)(1ULL << 53);
}
static float rng_normal(Rng *r, float mean, float std)
{
    double u1 = rng_uniform(r);
    double u2 = rng_uniform(r);
    if (u1 < 1e-300)
        u1 = 1e-300;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
    return (float)(mean + std * z);
}

/* =========================================================================
 * I/O
 * ====================================================================== */
int8_t *read_binary_data(const char *filename, size_t num_freq, size_t *out_size)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
        FATALF("Cannot open file: %s", filename);
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_size < 0)
        FATAL("ftell failed");
    if ((size_t)file_size % num_freq != 0)
        FATAL("File size not multiple of num_freq");
    int8_t *data = (int8_t *)malloc((size_t)file_size);
    if (!data)
        FATAL("malloc failed in read_binary_data");
    if (fread(data, 1, (size_t)file_size, f) != (size_t)file_size)
        FATAL("Error reading file");
    fclose(f);
    *out_size = (size_t)file_size;
    return data;
}

void write_binary_data(const char *filename, const int8_t *data, size_t data_size,
                       size_t nsamples, size_t nchans)
{
    FILE *f = fopen(filename, "wb");
    if (!f)
        FATALF("Cannot open file for writing: %s", filename);
    // if (data_size % nsamples != 0 || data_size % nchans != 0) FATALF("Data dimensions mismatch: %s", filename);
    if (fwrite(data, 1, data_size, f) != data_size)
        FATAL("Error writing file");
    fclose(f);
}

/* =========================================================================
 * Type conversions
 * ====================================================================== */
void int8_to_float(const int8_t *in, float *out, size_t N, float zero_off)
{
/* Pure element-wise transform — no accumulation, no ordering issue. */
#pragma omp parallel for schedule(static)
    for (size_t k = 0; k < N; ++k)
        out[k] = (float)in[k] - zero_off;
}

void float_to_int8(float *in, int8_t *out, size_t N, float outmean, float outstd)
{
    int max_threads = omp_get_max_threads();
    double *t_mean = (double *)calloc((size_t)max_threads, sizeof(double));
    double *t_sq = (double *)calloc((size_t)max_threads, sizeof(double));
    if (!t_mean || !t_sq)
        FATAL("malloc failed float_to_int8");

    int actual_threads = 1;

#pragma omp parallel shared(actual_threads)
    {
        int tid = omp_get_thread_num();
        double lmean = 0.0, lsq = 0.0;

#pragma omp for schedule(static) nowait
        for (size_t k = 0; k < N; ++k)
        {
            double v = in[k];
            lmean += v;
            lsq += v * v;
        }
        t_mean[tid] = lmean;
        t_sq[tid] = lsq;

#pragma omp single
        actual_threads = omp_get_num_threads();
    }

    /* Serial Kahan merge in fixed order — deterministic */
    KahanSum ks_mean = {0, 0}, ks_sq = {0, 0};
    for (int t = 0; t < actual_threads; ++t)
    {
        kahan_add(&ks_mean, t_mean[t]);
        kahan_add(&ks_sq, t_sq[t]);
    }
    free(t_mean);
    free(t_sq);

    double tmpmean = ks_mean.sum / (double)N;
    double tmpstd = ks_sq.sum / (double)N - tmpmean * tmpmean;
    tmpstd = sqrt(tmpstd);

    float scl = outstd / (float)tmpstd;
    float offs = outmean - scl * (float)tmpmean;

#pragma omp parallel for schedule(static)
    for (size_t k = 0; k < N; ++k)
    {
        float tmp = scl * in[k] + offs;
        tmp = roundf(tmp);
        tmp = clampf(tmp, 0.0f, 255.0f);
        out[k] = (int8_t)tmp;
    }
}

/* =========================================================================
 * Comparators
 * ====================================================================== */
static int cmp_float_asc(const void *a, const void *b)
{
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}
static int cmp_float_desc(const void *a, const void *b) { return -cmp_float_asc(a, b); }
static int cmp_double_asc(const void *a, const void *b)
{
    double fa = *(const double *)a, fb = *(const double *)b;
    return (fa > fb) - (fa < fb);
}
static int cmp_double_desc(const void *a, const void *b) { return -cmp_double_asc(a, b); }

/* =========================================================================
 * skf_filter — Statistical Kurtosis Filter
 * ====================================================================== */
void skf_filter(float *data, float thresig, size_t nsamples, size_t nchans)
{
    if (nsamples == 0 || nchans == 0)
        FATAL("Empty data in skf_filter");

    /* ---- Step 1: raw moments per channel (each channel independent) ---- */
    double *chmean1 = (double *)calloc(nchans, sizeof(double));
    double *chmean2 = (double *)calloc(nchans, sizeof(double));
    double *chmean3 = (double *)calloc(nchans, sizeof(double));
    double *chmean4 = (double *)calloc(nchans, sizeof(double));
    double *chcorr = (double *)calloc(nchans, sizeof(double));
    if (!chmean1 || !chmean2 || !chmean3 || !chmean4 || !chcorr)
        FATAL("malloc failed skf step1");

#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < nchans; ++j)
    {
        double m1 = 0, m2 = 0, m3 = 0, m4 = 0, corr = 0;
        double last = data[j];
        for (size_t i = 1; i < nsamples; ++i)
        {
            double v = data[i * nchans + j];
            double v2 = v * v;
            m1 += v;
            m2 += v2;
            m3 += v2 * v;
            m4 += v2 * v2;
            corr += v * last;
            last = v;
        }
        chmean1[j] = m1;
        chmean2[j] = m2;
        chmean3[j] = m3;
        chmean4[j] = m4;
        chcorr[j] = corr;
    }

    /* ---- Step 2: derived statistics (each channel independent) ---- */
    float *chmean = (float *)calloc(nchans, sizeof(float));
    float *chstd = (float *)calloc(nchans, sizeof(float));
    float *chskewness = (float *)calloc(nchans, sizeof(float));
    float *chkurtosis = (float *)calloc(nchans, sizeof(float));
    if (!chmean || !chstd || !chskewness || !chkurtosis)
        FATAL("malloc failed skf step2");

#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < nchans; ++j)
    {
        chmean1[j] /= nsamples;
        chmean2[j] /= nsamples;
        chmean3[j] /= nsamples;
        chmean4[j] /= nsamples;
        chcorr[j] /= (nsamples - 1);

        double mu = chmean1[j];
        double mu2 = mu * mu;
        double var = chmean2[j] - mu2;
        chmean[j] = (float)mu;

        if (var > 0.0)
        {
            chskewness[j] = (float)(chmean3[j] - 3.0 * chmean2[j] * mu + 2.0 * mu2 * mu);
            chkurtosis[j] = (float)(chmean4[j] - 4.0 * chmean3[j] * mu + 6.0 * chmean2[j] * mu2 - 3.0 * mu2 * mu2);
            chkurtosis[j] /= (float)(var * var);
            chkurtosis[j] -= 3.0f;
            chskewness[j] /= (float)(var * sqrt(var));
            chcorr[j] -= mu2;
            chcorr[j] /= var;
            chstd[j] = sqrtf((float)var);
        }
        else
        {
            chstd[j] = 1.0f;
            chkurtosis[j] = FLT_MAX;
            chskewness[j] = FLT_MAX;
            chcorr[j] = FLT_MAX;
        }
    }

    /* ---- Step 3: IQR  (serial sort on 4096 elements — negligible) ---- */
    size_t q_idx = nchans / 4;
    float kurtosis_q1, kurtosis_q3, kurtosis_R;
    float skewness_q1, skewness_q3, skewness_R;
    double corr_q1, corr_q3, corr_R;
    {
        float *tmp = (float *)malloc(nchans * sizeof(float));
        memcpy(tmp, chkurtosis, nchans * sizeof(float));
        qsort(tmp, nchans, sizeof(float), cmp_float_asc);
        kurtosis_q1 = tmp[q_idx];
        qsort(tmp, nchans, sizeof(float), cmp_float_desc);
        kurtosis_q3 = tmp[q_idx];
        kurtosis_R = kurtosis_q3 - kurtosis_q1;
        memcpy(tmp, chskewness, nchans * sizeof(float));
        qsort(tmp, nchans, sizeof(float), cmp_float_asc);
        skewness_q1 = tmp[q_idx];
        qsort(tmp, nchans, sizeof(float), cmp_float_desc);
        skewness_q3 = tmp[q_idx];
        skewness_R = skewness_q3 - skewness_q1;
        free(tmp);
    }
    {
        double *tmp = (double *)malloc(nchans * sizeof(double));
        for (size_t j = 0; j < nchans; ++j)
            tmp[j] = chcorr[j];
        qsort(tmp, nchans, sizeof(double), cmp_double_asc);
        corr_q1 = tmp[q_idx];
        qsort(tmp, nchans, sizeof(double), cmp_double_desc);
        corr_q3 = tmp[q_idx];
        corr_R = corr_q3 - corr_q1;
        free(tmp);
    }

    /* ---- Step 4: flag channels (each channel independent) ---- */
    int8_t *weights = (int8_t *)calloc(nchans, sizeof(int8_t));
    if (!weights)
        FATAL("malloc failed skf step4");

    long kill_count = 0;
    if (thresig >= 0)
    {
#pragma omp parallel for reduction(+ : kill_count) schedule(static)
        for (size_t j = 0; j < nchans; ++j)
        {
            if (chkurtosis[j] >= kurtosis_q1 - thresig * kurtosis_R &&
                chkurtosis[j] <= kurtosis_q3 + thresig * kurtosis_R &&
                chskewness[j] >= skewness_q1 - thresig * skewness_R &&
                chskewness[j] <= skewness_q3 + thresig * skewness_R &&
                chcorr[j] >= corr_q1 - thresig * corr_R &&
                chcorr[j] <= corr_q3 + thresig * corr_R)
            {
                weights[j] = 1;
            }
            else
            {
                ++kill_count;
            }
        }
    }
    else
    {
#pragma omp parallel for schedule(static)
        for (size_t j = 0; j < nchans; ++j)
            weights[j] = 1;
    }

#pragma omp critical
    printf("SKF: Kill rate = %.4f%%\n",
           (float)kill_count / (float)nchans * 100.0f);

/* ---- Step 5: normalise — each row independent ---- */
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nsamples; ++i)
    {
        float *row = &data[i * nchans];
        for (size_t j = 0; j < nchans; ++j)
            row[j] = (float)weights[j] * (row[j] - chmean[j]) / chstd[j];
    }

/* ---- Step 6: bad-channel fill — thread-local RNG ---- */
#pragma omp parallel
    {
        Rng rng;
        rng_init(&rng, omp_get_thread_num());
#pragma omp for schedule(static)
        for (size_t i = 0; i < nsamples; ++i)
            for (size_t j = 0; j < nchans; ++j)
                if (weights[j] == 0)
                    data[i * nchans + j] = rng_normal(&rng, 0.0f, 1.0f);
    }

    free(chmean1);
    free(chmean2);
    free(chmean3);
    free(chmean4);
    free(chcorr);
    free(chmean);
    free(chstd);
    free(chskewness);
    free(chkurtosis);
    free(weights);
}

/* =========================================================================
 * patch_filter
 *
 * The atomic neighbour writes are still safe: two threads writing 1 to
 * the same location is idempotent and atomic write prevents torn reads.
 * ====================================================================== */
void patch_filter(float *data, size_t nsamples, size_t nchans, int filltype)
{
    int *mask = (int *)calloc(nsamples, sizeof(int));
    if (!mask)
        FATAL("malloc failed patch_filter mask");

/* ---- Step 1: detect zero-variance rows ---- */
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nsamples; ++i)
    {
        double sum = 0.0, sq_sum = 0.0;
        for (size_t j = 0; j < nchans; ++j)
        {
            double v = data[i * nchans + j];
            sum += v;
            sq_sum += v * v;
        }
        double mean = sum / nchans;
        double var = sq_sum / nchans - mean * mean;
        if (var == 0.0)
        {
#pragma omp atomic write
            mask[i] = 1;
            if (i != 0)
            {
#pragma omp atomic write
                mask[i - 1] = 1;
            }
            if (i != nsamples - 1)
            {
#pragma omp atomic write
                mask[i + 1] = 1;
            }
        }
    }

    long kill_count = 0;
    for (size_t i = 0; i < nsamples; ++i)
        kill_count += mask[i];

#pragma omp critical
    printf("Patch Filter: Kill rate = %.6f\n",
           (double)kill_count / (double)nsamples);

    /* ---- Step 2: mean/var of non-flagged samples ----
     * NOTE: Original C++ bug preserved — `continue` always fires here,
     * so count stays 0 and chmean_patch/chvar_patch stay zero.
     * See original conversion comment for details. */
    double *chmean_patch = (double *)calloc(nchans, sizeof(double));
    double *chvar_patch = (double *)calloc(nchans, sizeof(double));
    if (!chmean_patch || !chvar_patch)
        FATAL("malloc failed patch step2");

    long count = 0;
    for (size_t i = 0; i < nsamples; ++i)
    {
        if (mask[i])
            printf("This time sample is flagged: %zu\n", i);
        continue; /* faithfully reproduces original bug */

        for (size_t j = 0; j < nchans; ++j)
        {
            chmean_patch[j] += data[i * nchans + j];
            chvar_patch[j] += data[i * nchans + j] * data[i * nchans + j];
        }
        ++count;
    }
    if (count > 0)
    {
        for (size_t j = 0; j < nchans; ++j)
        {
            chmean_patch[j] /= count;
            chvar_patch[j] = chvar_patch[j] / count - chmean_patch[j] * chmean_patch[j];
        }
    }

    /* ---- Step 3: fill ---- */
    if (filltype == 0)
    { /* mean fill */
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nsamples; ++i)
        {
            if (!mask[i])
                continue;
            for (size_t j = 0; j < nchans; ++j)
                data[i * nchans + j] = (float)chmean_patch[j];
        }
    }
    else
    { /* rand fill — thread-local RNG */
        float *ch_std = (float *)malloc(nchans * sizeof(float));
        if (!ch_std)
            FATAL("malloc failed patch rand std");
        for (size_t j = 0; j < nchans; ++j)
            ch_std[j] = (float)sqrt(chvar_patch[j] > 0.0 ? chvar_patch[j] : 0.0);

#pragma omp parallel
        {
            Rng rng;
            rng_init(&rng, omp_get_thread_num());
#pragma omp for schedule(static)
            for (size_t i = 0; i < nsamples; ++i)
            {
                if (!mask[i])
                    continue;
                for (size_t j = 0; j < nchans; ++j)
                    data[i * nchans + j] = rng_normal(&rng,
                                                      (float)chmean_patch[j],
                                                      ch_std[j]);
            }
        }
        free(ch_std);
    }

    free(mask);
    free(chmean_patch);
    free(chvar_patch);
}

/* =========================================================================
 * equalization
 * ====================================================================== */
void equalization(float *data, size_t nsamples, size_t nchans,
                  float *chmean, float *chstd)
{
    memset(chmean, 0, nchans * sizeof(float));
    memset(chstd, 0, nchans * sizeof(float));

    int max_threads = omp_get_max_threads();

    /* double scratch: avoids float precision loss during accumulation */
    double *acc_mean = (double *)calloc((size_t)max_threads * nchans, sizeof(double));
    double *acc_sq = (double *)calloc((size_t)max_threads * nchans, sizeof(double));
    if (!acc_mean || !acc_sq)
        FATAL("malloc failed equalization");

    int actual_threads = 1;

#pragma omp parallel shared(actual_threads)
    {
        int tid = omp_get_thread_num();
        double *lmean = acc_mean + (size_t)tid * nchans;
        double *lsq = acc_sq + (size_t)tid * nchans;

#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < nsamples; ++i)
        {
            const float *row = &data[i * nchans];
            for (size_t j = 0; j < nchans; ++j)
            {
                lmean[j] += (double)row[j];
                lsq[j] += (double)row[j] * row[j];
            }
        }

#pragma omp single
        actual_threads = omp_get_num_threads();
    }

    double *dmean = (double *)calloc(nchans, sizeof(double));
    double *dsq = (double *)calloc(nchans, sizeof(double));
    double *cmean_c = (double *)calloc(nchans, sizeof(double)); /* Kahan compensators */
    double *csq_c = (double *)calloc(nchans, sizeof(double));
    if (!dmean || !dsq || !cmean_c || !csq_c)
        FATAL("malloc failed equalization merge");

    for (int t = 0; t < actual_threads; ++t)
    {
        double *lmean = acc_mean + (size_t)t * nchans;
        double *lsq = acc_sq + (size_t)t * nchans;
        for (size_t j = 0; j < nchans; ++j)
        {
            /* Kahan add for dmean[j] */
            double y, tmp;
            y = lmean[j] - cmean_c[j];
            tmp = dmean[j] + y;
            cmean_c[j] = (tmp - dmean[j]) - y;
            dmean[j] = tmp;
            /* Kahan add for dsq[j] */
            y = lsq[j] - csq_c[j];
            tmp = dsq[j] + y;
            csq_c[j] = (tmp - dsq[j]) - y;
            dsq[j] = tmp;
        }
    }
    free(acc_mean);
    free(acc_sq);
    free(cmean_c);
    free(csq_c);

    for (size_t j = 0; j < nchans; ++j)
    {
        double m = dmean[j] / (double)nsamples;
        double s2 = dsq[j] / (double)nsamples - m * m;
        chmean[j] = (float)m;
        chstd[j] = (float)sqrt(s2 > 0.0 ? s2 : 0.0);
        if (chstd[j] == 0.0f)
            chstd[j] = 1.0f;
    }
    free(dmean);
    free(dsq);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nsamples; ++i)
    {
        float *row = &data[i * nchans];
        for (size_t j = 0; j < nchans; ++j)
            row[j] = (row[j] - chmean[j]) / chstd[j];
    }
}

/* =========================================================================
 * Heap helpers for sliding_median  (unchanged — serial, correct)
 * ====================================================================== */
typedef struct
{
    float *arr;
    size_t size, cap;
    int is_max;
} Heap;

static void heap_init(Heap *h, size_t cap, int is_max)
{
    h->arr = (float *)malloc(cap * sizeof(float));
    h->size = 0;
    h->cap = cap;
    h->is_max = is_max;
}
static void heap_free_h(Heap *h)
{
    free(h->arr);
    h->size = 0;
}
static int heap_cmp(const Heap *h, float a, float b)
{
    return h->is_max ? (a > b) : (a < b);
}
static void heap_push(Heap *h, float v)
{
    size_t i = h->size++;
    h->arr[i] = v;
    while (i > 0)
    {
        size_t p = (i - 1) / 2;
        if (heap_cmp(h, h->arr[i], h->arr[p]))
        {
            float t = h->arr[i];
            h->arr[i] = h->arr[p];
            h->arr[p] = t;
            i = p;
        }
        else
            break;
    }
}
static float heap_top(const Heap *h) { return h->arr[0]; }
static void heap_pop(Heap *h)
{
    h->arr[0] = h->arr[--h->size];
    size_t i = 0;
    for (;;)
    {
        size_t l = 2 * i + 1, r = 2 * i + 2, best = i;
        if (l < h->size && heap_cmp(h, h->arr[l], h->arr[best]))
            best = l;
        if (r < h->size && heap_cmp(h, h->arr[r], h->arr[best]))
            best = r;
        if (best == i)
            break;
        float t = h->arr[i];
        h->arr[i] = h->arr[best];
        h->arr[best] = t;
        i = best;
    }
}
static void heap_remove(Heap *h, float v)
{
    for (size_t i = 0; i < h->size; ++i)
    {
        if (h->arr[i] == v)
        {
            h->arr[i] = h->arr[--h->size];
            while (i > 0)
            {
                size_t p = (i - 1) / 2;
                if (heap_cmp(h, h->arr[i], h->arr[p]))
                {
                    float t = h->arr[i];
                    h->arr[i] = h->arr[p];
                    h->arr[p] = t;
                    i = p;
                }
                else
                    break;
            }
            size_t j = i;
            for (;;)
            {
                size_t l = 2 * j + 1, r = 2 * j + 2, best = j;
                if (l < h->size && heap_cmp(h, h->arr[l], h->arr[best]))
                    best = l;
                if (r < h->size && heap_cmp(h, h->arr[r], h->arr[best]))
                    best = r;
                if (best == j)
                    break;
                float t = h->arr[j];
                h->arr[j] = h->arr[best];
                h->arr[best] = t;
                j = best;
            }
            return;
        }
    }
}
static void heap_rebalance(Heap *low, Heap *high)
{
    if (low->size > high->size + 1)
    {
        heap_push(high, heap_top(low));
        heap_pop(low);
    }
    else if (high->size > low->size + 1)
    {
        heap_push(low, heap_top(high));
        heap_pop(high);
    }
}
static float heap_median(const Heap *low, const Heap *high)
{
    if (low->size > high->size)
        return heap_top(low);
    if (high->size > low->size)
        return heap_top(high);
    return (heap_top(low) + heap_top(high)) / 2.0f;
}

static void sliding_median_float(const float *data, float *out, long size, int w)
{
    if (w > (int)size)
        w = (int)size;
    Heap low, high;
    heap_init(&low, (size_t)w + 2, 1);
    heap_init(&high, (size_t)w + 2, 0);
    int a = -w / 2 - 1, b = (w - 1) / 2;
    heap_push(&low, data[0]);
    float median = data[0];
    for (int i = 1; i < b && i < (int)size; ++i)
    {
        if (data[i] >= median)
            heap_push(&high, data[i]);
        else
            heap_push(&low, data[i]);
        heap_rebalance(&low, &high);
        median = heap_median(&low, &high);
    }
    for (int i = 0; i < w / 2 + 1 && b < (int)size; ++i)
    {
        if (data[b] >= median)
            heap_push(&high, data[b]);
        else
            heap_push(&low, data[b]);
        heap_rebalance(&low, &high);
        median = heap_median(&low, &high);
        out[i] = median;
        ++a;
        ++b;
    }
    for (int i = w / 2 + 1; i < (int)size - (w - 1) / 2; ++i)
    {
        if (data[b] >= median)
            heap_push(&high, data[b]);
        else
            heap_push(&low, data[b]);
        int fl = 0;
        for (size_t k = 0; k < low.size; ++k)
            if (low.arr[k] == data[a])
            {
                fl = 1;
                break;
            }
        if (fl)
            heap_remove(&low, data[a]);
        else
            heap_remove(&high, data[a]);
        heap_rebalance(&low, &high);
        median = heap_median(&low, &high);
        out[i] = median;
        ++a;
        ++b;
    }
    for (int i = (int)size - (w - 1) / 2; i < (int)size; ++i)
    {
        int fl = 0;
        for (size_t k = 0; k < low.size; ++k)
            if (low.arr[k] == data[a])
            {
                fl = 1;
                break;
            }
        if (fl)
            heap_remove(&low, data[a]);
        else
            heap_remove(&high, data[a]);
        heap_rebalance(&low, &high);
        median = heap_median(&low, &high);
        out[i] = median;
        ++a;
    }
    heap_free_h(&low);
    heap_free_h(&high);
}

static void sliding_median_double(const double *data, double *out, long size, int w)
{
    float *fi = (float *)calloc((size_t)size, sizeof(float));
    float *fo = (float *)malloc((size_t)size * sizeof(float));
    if (!fi || !fo)
        FATAL("malloc failed sliding_median_double");
    for (long i = 0; i < size; ++i)
        fi[i] = (float)data[i];
    sliding_median_float(fi, fo, size, w);
    for (long i = 0; i < size; ++i)
        out[i] = fo[i];
    free(fi);
    free(fo);
}

/* =========================================================================
 * baseline_filter
 * ====================================================================== */
void baseline_filter(float *data, size_t nsamples, size_t nchans,
                     float width, float tsamp)
{
    double *s = (double *)calloc(nsamples, sizeof(double));
    if (!s)
        FATAL("malloc failed baseline s");

    int window_size = (int)(width / tsamp);
    int max_threads = omp_get_max_threads();

/* ---- Step 1: channel mean per time sample (each row independent) ---- */
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nsamples; ++i)
    {
        double sum = 0.0;
        const float *row = &data[i * nchans];
        for (size_t j = 0; j < nchans; ++j)
            sum += row[j];
        s[i] = sum / nchans;
    }

    /* ---- Step 2: sliding median (inherently serial) ---- */
    double *s_med = (double *)malloc(nsamples * sizeof(double));
    if (!s_med)
        FATAL("malloc failed baseline s_med");
    sliding_median_double(s, s_med, (long)nsamples, window_size);
    memcpy(s, s_med, nsamples * sizeof(double));
    free(s_med);

    double *t_xe = (double *)calloc((size_t)max_threads * nchans, sizeof(double));
    double *t_xs = (double *)calloc((size_t)max_threads * nchans, sizeof(double));
    double *t_se = (double *)calloc((size_t)max_threads, sizeof(double));
    double *t_ss = (double *)calloc((size_t)max_threads, sizeof(double));
    if (!t_xe || !t_xs || !t_se || !t_ss)
        FATAL("malloc failed baseline regression");

    int actual_threads = 1;

#pragma omp parallel shared(actual_threads)
    {
        int tid = omp_get_thread_num();
        double *lxe = t_xe + (size_t)tid * nchans;
        double *lxs = t_xs + (size_t)tid * nchans;
        double lse = 0.0, lss = 0.0;

#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < nsamples; ++i)
        {
            const float *row = &data[i * nchans];
            double si = s[i];
            for (size_t j = 0; j < nchans; ++j)
            {
                lxe[j] += (double)row[j];
                lxs[j] += (double)row[j] * si;
            }
            lse += si;
            lss += si * si;
        }
        t_se[tid] = lse;
        t_ss[tid] = lss;

#pragma omp single
        actual_threads = omp_get_num_threads();
    }

    double *xe = (double *)calloc(nchans, sizeof(double));
    double *xs = (double *)calloc(nchans, sizeof(double));
    double *cxe = (double *)calloc(nchans, sizeof(double)); /* compensators */
    double *cxs = (double *)calloc(nchans, sizeof(double));
    if (!xe || !xs || !cxe || !cxs)
        FATAL("malloc failed baseline merge");

    double se = 0.0, ss = 0.0, cse = 0.0, css = 0.0;
    for (int t = 0; t < actual_threads; ++t)
    {
        double *lxe = t_xe + (size_t)t * nchans;
        double *lxs = t_xs + (size_t)t * nchans;
        /* Kahan merge for xe, xs per channel */
        for (size_t j = 0; j < nchans; ++j)
        {
            double y, tmp;
            y = lxe[j] - cxe[j];
            tmp = xe[j] + y;
            cxe[j] = (tmp - xe[j]) - y;
            xe[j] = tmp;
            y = lxs[j] - cxs[j];
            tmp = xs[j] + y;
            cxs[j] = (tmp - xs[j]) - y;
            xs[j] = tmp;
        }
        /* Kahan merge for se, ss */
        {
            double y = t_se[t] - cse;
            double tmp = se + y;
            cse = (tmp - se) - y;
            se = tmp;
        }
        {
            double y = t_ss[t] - css;
            double tmp = ss + y;
            css = (tmp - ss) - y;
            ss = tmp;
        }
    }
    free(t_xe);
    free(t_xs);
    free(t_se);
    free(t_ss);
    free(cxe);
    free(cxs);

    /* ---- Step 4: coefficients (serial, cheap) ---- */
    double denom = se * se - ss * (double)nsamples;
    double *alpha = (double *)calloc(nchans, sizeof(double));
    double *beta = (double *)calloc(nchans, sizeof(double));
    if (!alpha || !beta)
        FATAL("malloc failed baseline coeffs");

    if (denom != 0.0)
    {
        for (size_t j = 0; j < nchans; ++j)
        {
            alpha[j] = (xe[j] * se - xs[j] * (double)nsamples) / denom;
            beta[j] = (xs[j] * se - xe[j] * ss) / denom;
        }
    }

/* ---- Step 5: subtract baseline (each row independent) ---- */
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nsamples; ++i)
    {
        float *row = &data[i * nchans];
        double si = s[i];
        for (size_t j = 0; j < nchans; ++j)
            row[j] -= (float)(alpha[j] * si + beta[j]);
    }

    free(s);
    free(xe);
    free(xs);
    free(alpha);
    free(beta);
}

/* =========================================================================
 * Output path helper
 * ====================================================================== */

static void build_output_path(const char *in_file, char *out_file, size_t out_len)
{
    const char *last_sep = NULL;
    for (const char *p = in_file; *p; ++p)
        if (*p == '/' || *p == '\\')
            last_sep = p;
    const char *base = last_sep ? last_sep + 1 : in_file;
    size_t dir_len = (size_t)(base - in_file);
    if (dir_len >= out_len)
        FATAL("Path too long");
    const char *dot = NULL;
    for (const char *p = base; *p; ++p)
        if (*p == '.')
            dot = p;
    size_t stem_len = dot ? (size_t)(dot - base) : strlen(base);
    snprintf(out_file, out_len, "%.*s%.*s_stripe.raw",
             (int)dir_len, in_file, (int)stem_len, base);
}

/* =========================================================================
 * process_block
 * ====================================================================== */
static void process_block(int8_t *block_ptr, size_t block_N,
                          size_t nsamples, size_t nchans, int filltype,
                          float thresig, float outmean, float outstd,
                          float width, float tsamp)
{
    float *fd = (float *)malloc(block_N * sizeof(float));
    float *cmean = (float *)malloc(nchans * sizeof(float));
    float *cstd = (float *)malloc(nchans * sizeof(float));
    if (!fd || !cmean || !cstd)
        FATAL("malloc failed in process_block");

    int8_to_float(block_ptr, fd, block_N, 64.0f);
    patch_filter(fd, nsamples, nchans, filltype);
    skf_filter(fd, thresig, nsamples, nchans);
    equalization(fd, nsamples, nchans, cmean, cstd);
    baseline_filter(fd, nsamples, nchans, width, tsamp);
    float_to_int8(fd, block_ptr, block_N, outmean, outstd);

    free(fd);
    free(cmean);
    free(cstd);
}

/* =========================================================================
 * main
 * ====================================================================== */
int main(int argc, char *argv[])
{
    double t_start = now_msec();

    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s <binary_file_path> <block_size> <threshold> [num_threads]\n", argv[0]);
        return 1;
    }

    const char *in_file = argv[1];
    int block_size = atoi(argv[2]);
    float thresig = (float)atof(argv[3]);

    if (argc >= 5)
    {
        int nt = atoi(argv[4]);
        if (nt > 0)
            omp_set_num_threads(nt);
    }

    omp_set_num_threads(8); // hardcoded 8 thread, getting peak performance for 8 threads.
    printf("OpenMP threads: %d\n", omp_get_max_threads());

    char out_file[4096];
    build_output_path(in_file, out_file, sizeof(out_file));

    // take output path as argument, if provided
    // if (argc >= 6)
    // {
    //     strncpy(out_file, argv[5], sizeof(out_file) - 1);
    //     out_file[sizeof(out_file) - 1] = '\0'; // ensure null-termination
    // }

    const int filltype = 1;
    const int nchans = 4096;
    const float outmean = 64.0f; // vary this
    const float outstd = 3.0f;
    const float width = 0.5f; // vary this (low and high)
    const float tsamp = 1.31072e-3f;

    size_t raw_size = 0;
    int8_t *raw_data = read_binary_data(in_file, (size_t)nchans, &raw_size); // SHM data pointer to be *raw_data;

    size_t nsamples = (size_t)block_size;
    size_t block_len = nsamples * (size_t)nchans;
    size_t n_full = raw_size / block_len; // this will always be 1
    // size_t remainder = raw_size % block_len;

    omp_set_num_threads(16);
    omp_set_nested(0);
    omp_set_max_active_levels(1);

    // printf("Total blocks: %zu  |  Remainder bytes: %zu\n", n_full, remainder);

#pragma omp parallel for num_threads(16) schedule(static) shared(raw_data)
    for (size_t blk = 0; blk < n_full; ++blk)
    {
        int8_t *block_ptr = raw_data + blk * block_len;
        size_t blk_id = blk;

        double blk_start = now_usec();
#pragma omp critical
        printf("-----------Processing block %zu / %zu  [thread %d]\n", blk_id + 1, n_full, omp_get_thread_num());

        process_block(block_ptr, nsamples * (size_t)nchans,
                      nsamples, (size_t)nchans, filltype,
                      thresig, outmean, outstd, width, tsamp);

        double blk_time = now_usec() - blk_start;
        float rtf = (block_size * tsamp * 1e6f) / (float)blk_time;
#pragma omp critical
        printf("RTF block %zu: %.4f  (%.0f us)\n", blk_id + 1, rtf, blk_time);
    }

    fprintf(stderr, "RAW SIZE = %ld NSAMP = %d NCHANS = %d \n", raw_size, nsamples, nchans);
    write_binary_data(out_file, raw_data, raw_size, nsamples, (size_t)nchans);

    double total_ms = now_msec() - t_start;
    printf("\nTotal RFI mitigation time: %.2f ms\n", total_ms);
    printf("Total Real-Time Factor: %.4f\n",
           (float)(raw_size * tsamp * 1e3) / (4096.0f * (float)total_ms));

    free(raw_data);
    return 0;
}
