/*
 ============================================================================
 Name         : hi2.c
 Author       : Christopher Paolini
 Version      :
 Copyright    : 
 Description  : CompE596 OpenMP Example 1
 Compile/link : gcc -fopenmp hi.c -o hi
 Execute      : export OMP_NUM_THREADS=128
                ./hi2
 ============================================================================
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sched.h>
int sched_getcpu(void);

int main (int argc, char *argv[])
{
#pragma omp parallel
  {
    //printf("Hi from thread number %d\n", omp_get_thread_num());

    printf("OMP thread %03d/%03d mapped to hwthread %03d\n",
    	   omp_get_thread_num(), omp_get_num_threads(), sched_getcpu());
    /*    
#pragma omp master
    {
      printf("This only once %d\n", omp_get_thread_num());
    }
    */
  }
  return 0;
}
