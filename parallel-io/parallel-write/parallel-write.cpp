#include <vector>
#include <cstdio>
#include <string>

#include <mpi.h>


// How many integers to write, total from all MPI processes
static constexpr size_t numElements = 32;

/* Enables or disables debug printing of file contents. Printing is not practical for large files,
so we enable/disable this based on 'numElements'. */
static constexpr bool doDebugPrint = (numElements <= 100);


// Debugging helper, prints out file contents. You don't have to modify this
void debug_read_file(const char* filename);

void single_writer(const std::vector<int>& localData, const char* filename) {
    // Gets called from all MPI ranks. 'localData' contains different data on each rank.
    int rank, ntasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    // You can assume that 'localData' has same length in all MPI processes:
    const size_t numElementsPerRank = localData.size();
    
    // "Spokesperson strategy": Send all data to rank 0 and write it from there.
    std::vector<int> receiveBuffer(
        (rank == 0) ? ntasks * numElementsPerRank : 0
    );

    MPI_Gather(localData.data(), numElementsPerRank, MPI_INT,
               receiveBuffer.data(), numElementsPerRank, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE *fileptr = fopen(filename, "wb");
        if (fileptr == NULL) {
            fprintf(stderr, "Failed to open file %s for writing!\n", filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fwrite(receiveBuffer.data(), sizeof(int), receiveBuffer.size(), fileptr);
        fclose(fileptr);
    }
}

void collective_write(const std::vector<int>& localData, const char* filename) {
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

    int ntasks;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    // We assume that each rank has the same amount of data
    MPI_Offset fileSize = localData.size() * sizeof(int) * ntasks;
    MPI_File_set_size(file, fileSize);

    // Rank is the offset of where to write in the file.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    MPI_Offset writeOffset = (MPI_Offset) (rank * localData.size() * sizeof(int));

    MPI_File_write_at_all(file, writeOffset, localData.data(), localData.size(), MPI_INT, MPI_STATUS_IGNORE);

    MPI_File_close(&file);
    }

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, ntasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    if (numElements % ntasks != 0) {
        if (rank == 0) {
            fprintf(stderr, "numElements (%zu) must be divisible by the number of MPI tasks (%d)!\n", numElements, ntasks);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const size_t numElementsPerRank = numElements / ntasks;

    // Create data array, each element initialized to value 'rank'
    std::vector<int> localData(numElementsPerRank, rank);

    // Print some statistics
    if (rank == 0) {
        printf("Writing total of %zu integers, %zu from each rank.\n", numElements, numElementsPerRank);
        const size_t bytes = numElements * sizeof(int);
        printf("Total bytes to write: %zu (%zu MB)\n", bytes, bytes / 1024 / 1024);
        fflush(stdout);
    }

    const int repeatCount = 5; // How many times to repeat the write operations

    // ########## "Spokesperson" write
    std::string filename = "single_writer.dat";

    for (int i = 0; i < repeatCount; ++i) {
        // Start time measurement
        double startTime = MPI_Wtime();

        single_writer(localData, filename.c_str());
        double endTime = MPI_Wtime();
        double elapsedTime = endTime - startTime;
        if (rank == 0) {
            printf("[%s] elapsed time: %f seconds\n", filename.c_str(), elapsedTime);
        }
        if (rank == 0 && doDebugPrint) {
            printf("[%s] file contents:\n", filename.c_str());
            debug_read_file(filename.c_str());
        }
    }

    

    // ########## Collective write
    filename = "collective_write.dat";

    for (int i = 0; i < repeatCount; ++i) {
        // Start time measurement
        double startTime = MPI_Wtime();

        collective_write(localData, filename.c_str());
        double endTime = MPI_Wtime();
        double elapsedTime = endTime - startTime;
        if (rank == 0) {
            printf("[%s] elapsed time: %f seconds\n", filename.c_str(), elapsedTime);
        }
        if (rank == 0 && doDebugPrint) {
            printf("[%s] file contents:\n", filename.c_str());
            debug_read_file(filename.c_str());
        }
    }

    //~

    MPI_Finalize();
    return 0;
}


void debug_read_file(const char* filename) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        FILE *fileptr = fopen(filename, "rb");

        if (fileptr != NULL) {

            int value;
            while (fread(&value, sizeof(int), 1, fileptr) == 1) {
                printf("%d", value);
            }

            fclose(fileptr);

            printf("\n");
            fflush(stdout);
        } else {
            fprintf(stderr, "Failed to open file %s for debug printing!\n", filename);
        }
    }
}
