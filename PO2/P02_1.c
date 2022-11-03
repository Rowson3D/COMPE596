///
/// First iteration of P02
/// Solving the simpler integral with 0 to 1 boundaries
///

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to integrate
double f(const double x)
{
  return 1.0 / (1.0 + x * x);
}

int main (int argc, char *argv[])
{
  int numThreads = 1, tid = 0;   
  const double n = 1e9; // number of intervals must be even
  int N = (int)n/2;
  int istart = tid * N / numThreads;
  int iend = (tid + 1) * N / numThreads;
  if (tid == numThreads - 1) iend = N;
  const double b = 1;
  const double a = 0;
  const double h = (b - a) / n;
  double* s = (double*)calloc(numThreads, sizeof(double));
  double start_time = omp_get_wtime(), elapsedTime;
  for (unsigned int j = istart + 1; j <= iend; ++j)
   {
      s[tid] +=
              f(a + (2 * j - 2) * h) +
          4 * f(a + (2 * j - 1) * h) +
              f(a + (2 * j) * h);
  }
  s[tid] *= h / 3;
  elapsedTime = omp_get_wtime() - start_time;
  double exact = M_PI / 4;
  printf("approximation: %f, error: %e, intervals: %.0f, runtime: %f s, threads: %03d\n",
          s[tid], fabs(s[tid] - exact), n, elapsedTime, numThreads);
 return 0;
}