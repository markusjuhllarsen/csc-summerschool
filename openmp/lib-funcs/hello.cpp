#include <cstdio>
#include <omp.h>
int main()
{   
    int thread_id, num_threads;

    printf("Hello world!\n");
    #pragma omp parallel private(thread_id) shared(num_threads)
    {   
        #ifdef _OPENMP
        thread_id = omp_get_thread_num();

        #pragma omp single
        // This block is executed by only one thread
        num_threads = omp_get_num_threads();
        #else
        thread_id = 1; // Fallback for non-OpenMP environments
        num_threads = 1; // Fallback for non-OpenMP environments
        #endif

        #pragma omp critical
        // This block is executed by only one thread at a time
        // to not have interleaving output from different threads
        printf("Hello from thread %d\n", thread_id);
    }
    printf("There are %d threads in total.\n", num_threads);
    return 0;
}