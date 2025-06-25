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
    double half_pi = 0.0;
    for (int i=istart; i <= istop/2; i++) {
        double x = (i - 0.5) / n;
        half_pi += 1.0 / (1.0 + x*x);
    }

    double pi = half_pi;
    MPI_Recv(&half_pi, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    pi += half_pi;
    pi *= 4.0 / n;
    printf("Approximate pi=%18.16f (exact pi=%10.8f)\n", pi, M_PI);
}
  
  if (rank == 1){
    double half_pi = 0.0;
    for (int i=istop/2+1; i <= istop; i++) {
        double x = (i - 0.5) / n;
        half_pi += 1.0 / (1.0 + x*x);
    }
    MPI_Send(&half_pi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }


  MPI_Finalize();

}
