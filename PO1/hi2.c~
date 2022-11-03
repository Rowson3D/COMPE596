/*
 ============================================================================
 Name         : hi.c
 Author       : Christopher Paolini
 Version      :
 Copyright    : 
 Description  : CompE596 OpenMP Example 1
 Compile/link : gcc -fopenmp hi.c -o hi
 Execute      : export OMP_NUM_THREADS=128
                ./hi
 ============================================================================
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main (int argc, char *argv[])
{
#pragma omp parallel
  {
    printf("Hi from thread number %d\n", omp_get_thread_num());
  }
  return 0;
}
