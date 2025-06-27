#include <cstdio>

#define NX 102400

int main(void)
{
    long vecA[NX];
    long sum, psum, sumex;
    int i;

    /* Initialization of the vectors */
    for (i = 0; i < NX; i++) {
        vecA[i] = (long) i + 1;
    }

    sumex = (long) NX * (NX + 1) / ((long) 2);
    printf("Arithmetic sum formula (exact): %ld\n",sumex);
    
    sum = 0.0;
    #pragma omp parallel for private(i)
    for (i = 0; i < NX; i++) {
        sum += vecA[i];
    }
    printf("(Parallel) Sum: %ld\n", sum);

    sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (i = 0; i < NX; i++) {
        sum += vecA[i];
    }
    printf("(Reduction) Sum: %ld\n", sum);

    // Parallelize using partial sums and critical section
    sum = 0.0;
    psum = 0.0;
    #pragma omp parallel private(i) firstprivate(psum)
    {
        #pragma omp for
        for (i = 0; i < NX; i++) {
            psum += vecA[i];
        }
        #pragma omp critical
        sum += psum;
    }
    printf("(Critical) Sum: %ld\n", sum);

    return 0;
}
