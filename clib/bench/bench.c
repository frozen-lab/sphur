#include "../include/sphur.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_BINS 256
#define NUM_RANDOM 10000

// High-resolution timer
static double get_time_us() {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);

  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

// Benchmark throughput & store numbers in buffer
void benchmark_throughput(sphur_t *state, uint64_t *numbers, size_t n) {
  double start = get_time_us();

  for (size_t i = 0; i < n; i++) {
    numbers[i] = sphur_gen_rand(state); // generate 1 random number
  }

  double end = get_time_us();
  double elapsed_us = end - start;
  double nums_per_us = (double)n / elapsed_us;

  printf("Throughput: %.2f random numbers per microsecond\n", nums_per_us);
}

// Benchmark randomness: histogram & autocorrelation
void benchmark_randomness(uint64_t *numbers, size_t n) {
  // ▶ Histogram / uniformity
  int bins[NUM_BINS] = {0};

  for (size_t i = 0; i < n; i++)
    bins[numbers[i] % NUM_BINS]++;

  double expected = (double)n / NUM_BINS;
  double chi2 = 0.0;

  for (int i = 0; i < NUM_BINS; i++) {
    double diff = bins[i] - expected;
    chi2 += diff * diff / expected;
  }

  printf("Chi-squared (uniformity): %.2f\n", chi2);

  // ▶ Autocorrelation (lag 1)
  double mean = 0.0;

  for (size_t i = 0; i < n; i++)
    mean += numbers[i];

  mean /= n;

  double num = 0.0, den = 0.0;

  for (size_t i = 0; i < n - 1; i++) {
    num += (numbers[i] - mean) * (numbers[i + 1] - mean);
    den += (numbers[i] - mean) * (numbers[i] - mean);
  }

  double autocorr = num / den;
  printf("Autocorrelation (lag 1): %.5f\n", autocorr);
}

// Save numbers to file for Python visualization
void save_numbers_to_file(uint64_t *numbers, size_t n, const char *filepath) {
  FILE *f = fopen(filepath, "w");

  if (!f) {
    perror("Failed to open file");
    exit(1);
  }

  for (size_t i = 0; i < n; i++)
    fprintf(f, "%llu\n", (unsigned long long)numbers[i]);

  fclose(f);
  printf("Random numbers written to: %s\n", filepath);
}

int main() {
  printf("=== Sphur PRNG Benchmark ===\n");

  sphur_t sphur;
  uint64_t seed = 0x9e3779b97f4a7c15ULL;

  if (sphur_init_seeded(&sphur, seed) != 0) {
    printf("Failed to init sphur with seed\n");
    return 1;
  }

  uint64_t *numbers = malloc(NUM_RANDOM * sizeof(uint64_t));

  if (!numbers) {
    printf("Failed to allocate memory\n");
    return 1;
  }

  // Generate numbers & benchmark throughput
  benchmark_throughput(&sphur, numbers, NUM_RANDOM);

  // Benchmark randomness using the same numbers
  benchmark_randomness(numbers, NUM_RANDOM);

  // Save numbers for Python plotting
  save_numbers_to_file(numbers, NUM_RANDOM, "./build/prngs.txt");

  free(numbers);
  printf("Done ✅\n");

  return 0;
}
