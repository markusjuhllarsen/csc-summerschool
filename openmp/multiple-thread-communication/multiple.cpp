#include <cstdio>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    int rank, thread_id, ntasks, nthreads;
    int provided, required=MPI_THREAD_MULTIPLE;

    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        printf("MPI does not support MPI_THREAD_MULTIPLE\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
        return 0;
    }
    
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    #pragma omp parallel private(thread_id, nthreads)
    {
        thread_id = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        if (rank == 0) {
            MPI_Send(&thread_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        } else {
            int msg;
            MPI_Recv(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Rank %d, Thread %d received message from Rank 0: %d\n", rank, thread_id, msg);
        }
    }

    MPI_Finalize();
    return 0;
}
