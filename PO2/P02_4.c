#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to integrate
double f(const double x)
{
  return acos(cos(x) / (1 + (2* cos(x))));
}

int main (int argc, char *argv[])
{
  unsigned int Nthrds = 0;
  sscanf(argv[1],"%d",&Nthrds);
  printf("Number of threads requested: %d\n", Nthrds);
  
  omp_set_dynamic(0);
  omp_set_num_threads(Nthrds);
  
  double S = 0.0;
  int numThreads = Nthrds;
  double* s = (double*)calloc(numThreads, sizeof(double));
  
  double start_time = omp_get_wtime(), elapsedTime;
 
  double temp = 0;
  const double n = 1e8; // number of intervals must be even
  unsigned long long int N = (int)n/2;
  int tid;
  
#pragma omp parallel 
{
  unsigned long int istart = omp_get_thread_num() * N / numThreads;
  unsigned long int iend = (omp_get_thread_num() + 1) * N / numThreads;
  if (omp_get_thread_num() == numThreads - 1) iend = N;

  const double b = M_PI / 2;
  const double a = 0;
  const double h = (b - a) / n;

  for (unsigned int j = istart + 1; j <= iend; ++j)
   {
      s[omp_get_thread_num()] +=
              f(a + (2 * j - 2) * h) +
          4 * f(a + (2 * j - 1) * h) +
              f(a + (2 * j) * h);
  }
  s[omp_get_thread_num()] *= h / 3;
}
// Output all thread singular values
for (int i = 0; i < numThreads; i++)
{
    printf("s[%d] = %f\n", i, s[i]);
    // Store value so we can add it to the total sum
    temp = s[i];
    // Add the value to the total sum
    S += temp;
}

//Now print the total sum
printf("Total sum = %f\n", S);



#pragma omp parallel reduction(+:S)
   S = s[omp_get_thread_num()];
  
  elapsedTime = omp_get_wtime() - start_time;
  double exact = (M_PI * M_PI * 5) / 24;
  printf("approximation: %f, error: %e, intervals: %.0f, runtime: %f s, threads: %03d\n",
          S, fabs(S - exact), n, elapsedTime, numThreads);
 return 0;
}
