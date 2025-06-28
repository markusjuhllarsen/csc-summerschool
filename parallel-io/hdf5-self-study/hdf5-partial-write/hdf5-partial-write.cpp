#include <hdf5.h> // C-style HDF5 API
#include <mpi.h>

void partialWrite1D(const hid_t fileId) {

    // Create a 1D dataspace of length 10 and a corresponding dataset
    const hsize_t dims[1] = { 10 };
    hid_t dataspace = H5Screate_simple(1, dims, NULL);

    hid_t dataset = H5Dcreate(fileId, "MyDataset1D", H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Now each element in the dataset is 0 (default fill value in HDF5)


    // Create a *smaller* array that is to be written to the dataset
    const size_t dataLength = 8;
    int data[dataLength];
    // Give some nonzero values so that we can distinguish them from the default fill value
    for (size_t i = 0; i < dataLength; i++) {
        data[i] = (int) (i + 1);
    }

    // TODO: Write the 'data' array into the dataset. Ensure only the first 8 elements of the dataset are written to.
    herr_t status = H5Dwrite(
        dataset, 
        H5T_NATIVE_INT, 
        dataspace, 
        dataspace, 
        H5P_DEFAULT, 
        data
    );
    H5Dclose(dataset);
    H5Sclose(dataspace);
}

void partialWrite2D(const hid_t fileId) {

    // Create a 2D dataspace and a corresponding dataset
    const hsize_t dims[2] = { 6, 6 };
    hid_t dataspace = H5Screate_simple(2, dims, NULL);

    hid_t dataset = H5Dcreate(fileId, "MyDataset2D", H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Create a 2x3 array, implemented as 1D for simplicity
    const size_t rows = 2;
    const size_t cols = 3;
    int data[rows * cols];

    // Give some nonzero values
    for (size_t i = 0; i < rows * cols; i++) {
        data[i] = (int) (i+1);
    }

    // The resulting dataset should have its top-left and bottom-right block replaced with 'data' contents
    hsize_t memspaceDims[2] = { rows, cols };
    hid_t memspace = H5Screate_simple(2, memspaceDims, NULL);

    // Top left block of the dataset to write to
    // We will write a 2x3 block starting from the top-left corner of the dataset
    // The offset is the starting point in the dataset where we will write the block
    // The stride is how many elements to skip in each dimension when writing the block
    // The block is the size of the block to write, and count is how many blocks
    // we want to write (in this case, just one block)
    hsize_t offset[2] = { 0, 0 }; // Start writing at the top-left corner
    hsize_t stride[2] = { 1, 1 }; // Write every element in the block
    hsize_t block[2] = { rows, cols }; // Block size to write
    hsize_t count[2] = { 1, 1 }; // Write a 2x3 block

    // Select the hyperslab in the file
    herr_t status = H5Sselect_hyperslab(
        dataspace, 
        H5S_SELECT_SET, 
        offset, 
        stride, 
        count, 
        block    // TODO: write the 'data' array to the dataset as instructed in the exercise README.md.

    );

    // Offset to bottom-right corner of the dataset
    hsize_t offset[2] = {4,3};
    status = H5Sselect_hyperslab(
        dataspace,
        H5S_SELECT_SET,
        offset,
        stride,
        count,
        block
    );

    // Write the data
    status = H5Dwrite(dataset, H5T_NATIVE_INT, memspace, dataspace, H5P_DEFAULT, data);

    // Close resources
    H5Sclose(memspace);
    H5Dclose(dataset);
    H5Sclose(dataspace);
}

int main(int argc, char** argv) {

    /* This program is serial and is not intended to be ran using MPI.
    Accidentally running with many MPI processes would lead to competing HDF5 writes and other mess, so to prevent this
    we still initialize MPI here and ensure only one rank does the work.
    */
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {

        // Create an HDF5 file, truncating existing contents if the file already exists
        hid_t fileId = H5Fcreate(
            "partial_write.h5",
            H5F_ACC_TRUNC,
            H5P_DEFAULT,
            H5P_DEFAULT
        );

        // ### Part 1 ###
        partialWrite1D(fileId);

        // ### Part 2 ###
        partialWrite2D(fileId);

        // Close the file
        H5Fclose(fileId);
    }

    MPI_Finalize();
    return 0;
}
