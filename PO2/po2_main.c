#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to integrate
double f(const double x)
{
	return acos(cos(x) / (1 + (2* cos(x))));
}

int main(int argc, char *argv[])
{

	unsigned int Nthrds = 0;
	sscanf(argv[1], "%d", &Nthrds);
	printf("Number of threads requested: %d \n", Nthrds);

	omp_set_dynamic(0);
	omp_set_num_threads(Nthrds);
 
	double S;
	int numThreads = Nthrds;
	double *s = (double*) calloc(numThreads, sizeof(double));

	double temp = 0;
	const double n = 1e9;	// number of intervals must be even
	int N = (int) n / 2;
	double run_time;
	double start_time = omp_get_wtime(), elapsedTime;
	int tid;
 
  #pragma omp parallel
	{
		tid = omp_get_thread_num();
		int istart = tid *N / numThreads;
		int iend = (tid + 1) *N / numThreads;
		if (tid == numThreads - 1) iend = N;
		const double b = M_PI / 2;
		const double a = 0;
		const double h = (b - a) / n;

		for (unsigned int j = istart + 1; j <= iend; ++j)
		{
			temp =
				f(a + (2 *j - 2) *h) +
				4* f(a + (2 *j - 1) *h) +
				f(a + (2 *j) *h);

			s[tid] += temp;
		}
		s[tid] *= h / 3;
	}

	#pragma omp parallel reduction(+: S)
	S = s[omp_get_thread_num()];

	elapsedTime = omp_get_wtime() - start_time;
	double exact = (M_PI *M_PI *5) / 24;
	printf("approximation: %f, error: %e, intervals: %.0f, runtime: %f s, threads: %03d\n",
		s[tid], fabs(s[tid] - exact), n, elapsedTime, numThreads);
	return 0;
}