#include <cstdio>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    int rank, thread_id;
    int provided, required=MPI_THREAD_FUNNELED;

    MPI_Init_thread(&argc, &argv, required, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    #pragma omp parallel private(thread_id)
    {
        thread_id = omp_get_thread_num();
        printf("I'm thread %d in process %d\n", thread_id, rank);
    }

    /* TODO: Investigate the provided thread support level. */

    MPI_Finalize();
    return 0;
}
