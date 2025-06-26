#include <stdio.h>
#include <math.h>
#include <mpi.h>

const int n = 840;

int main(int argc, char** argv)
{
  
  int rank, ntasks;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int istart = n / ntasks * rank + 1;
  int istop = n / ntasks * rank + n / ntasks;

  int remainder = n % ntasks;
  // Adjust the start and stop indices if there are any remaining elements
  // that need to be distributed among the processes.
  // The first 'remainder' processes will get one extra element.
  // The last 'ntasks - remainder' processes will get the same number of elements.
  if (remainder > 0){
    if (rank < remainder){
      istart += rank;
      istop += rank + 1;
    } else {
      istart += remainder;
      istop += remainder;
    }
  }

  if (0 == rank) {
    printf("Computing approximation to pi with N=%d\n", n);
    printf("Using %d MPI processes\n", ntasks);
  }
  
  double local_pi = 0.0;
    for (int i=istart; i <= istop; i++) {
        double x = (i - 0.5) / n;
        local_pi += 1.0 / (1.0 + x*x);
  }

  double pi;
  MPI_Reduce(&local_pi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0){
    pi *= 4.0 / n;
    printf("Approximate pi=%18.16f (exact pi=%10.8f)\n", pi, M_PI);
  }


  MPI_Finalize();

}
