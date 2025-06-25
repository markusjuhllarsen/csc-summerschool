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

  int istart = 1;
  int istop = n;

   if (0 == rank) {
      printf("Computing approximation to pi with N=%d\n", n);
      printf("Using %d MPI processes\n", ntasks);
   }
  
  if (rank == 0){
    double local_pi = 0.0;
    for (int i=istart; i <= istop/2; i++) {
        double x = (i - 0.5) / n;
        local_pi += 1.0 / (1.0 + x*x);
    }

    double pi = local_pi;
    for (int i=1; i < ntasks; i++) {
        MPI_Recv(&local_pi, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pi += local_pi;
    }
    pi *= 4.0 / n;
    printf("Approximate pi=%18.16f (exact pi=%10.8f)\n", pi, M_PI);
  }
  
  if (rank != 0){
    double local_pi = 0.0;
    istart = n / ntasks * rank + 1;
    istop = n / ntasks * rank + n / ntasks;
    printf("Process %d computing from %d to %d\n", rank, istart, istop);
    for (int i=istart; i <= istop; i++) {
        double x = (i - 0.5) / n;
        local_pi += 1.0 / (1.0 + x*x);
    }
    MPI_Send(&local_pi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }


  MPI_Finalize();

}
