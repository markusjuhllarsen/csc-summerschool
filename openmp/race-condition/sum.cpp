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

    return 0;
}
