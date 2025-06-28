#include <array>

#include <hdf5.h> // C-style HDF5 API
#include <mpi.h>

void writeDataset1D(const std::array<int, 16>& arrayToWrite, const hid_t fileId, const char* datasetName) {    hsize_t dims[1] = {arrayToWrite.size()}; // Dimensions of the dataset
    // Create a dataspace for the dataset
    hid_t dataspace_id = H5Screate_simple(
        1, 
        dims, 
        nullptr
    );

    // Create the dataset in the file
    hid_t dataset_id = H5Dcreate(
        fileId, 
        datasetName, 
        H5T_NATIVE_INT, // Data type
        dataspace_id, 
        H5P_DEFAULT, 
        H5P_DEFAULT, 
        H5P_DEFAULT
    );

    //Write the data to the dataset
    herr_t status = H5Dwrite(
        dataset_id, 
        H5T_NATIVE_INT, // Data type
        H5S_ALL,        // Memory space
        H5S_ALL,        // File space
        H5P_DEFAULT,    // Transfer properties
        arrayToWrite.data() // Pointer to the data to write
    );

    // Close the dataset and dataspace
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
}

void writeDataset2D(const std::array<int, 16>& arrayToWrite, const hid_t fileId, const char* datasetName) {
    hsize_t dims[2] = {4, 4}; // Dimensions of the dataset
    // Create a dataspace for the dataset
    hid_t dataspace_id = H5Screate_simple(
        2, 
        dims, 
        nullptr
    );

    // Create the dataset in the file
    hid_t dataset_id = H5Dcreate(
        fileId, 
        datasetName, 
        H5T_NATIVE_INT, // Data type
        dataspace_id, 
        H5P_DEFAULT, 
        H5P_DEFAULT, 
        H5P_DEFAULT
    );

    //Write the data to the dataset
    herr_t status = H5Dwrite(
        dataset_id, 
        H5T_NATIVE_INT, // Data type
        H5S_ALL,        // Memory space
        H5S_ALL,        // File space
        H5P_DEFAULT,    // Transfer properties
        arrayToWrite.data() // Pointer to the data to write
    );

    // Close the dataset and dataspace
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
}

void writeDataset3D(const std::array<int, 16>& arrayToWrite, const hid_t fileId, const char* datasetName) {
    hsize_t dims[3] = {2, 2, 4}; // Dimensions of the dataset
    // Create a dataspace for the dataset
    hid_t dataspace_id = H5Screate_simple(
        3, 
        dims, 
        nullptr
    );

    // Create the dataset in the file
    hid_t dataset_id = H5Dcreate(
        fileId, 
        datasetName, 
        H5T_NATIVE_INT, // Data type
        dataspace_id, 
        H5P_DEFAULT, 
        H5P_DEFAULT, 
        H5P_DEFAULT
    );

    //Write the data to the dataset
    herr_t status = H5Dwrite(
        dataset_id, 
        H5T_NATIVE_INT, // Data type
        H5S_ALL,        // Memory space
        H5S_ALL,        // File space
        H5P_DEFAULT,    // Transfer properties
        arrayToWrite.data() // Pointer to the data to write
    );

    // Close the dataset and dataspace
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
}


// For part 2 of the exercise
void writeAttribute(const char* attributeName, double attributeValue, hid_t fileId, const char* datasetName) {

    // TODO: write a 'double' attribute of given name and value to the specified file and dataset
    // Remember to close any HDF5 resources you open here
    hid_t attributespace_id = H5Screate(H5S_SCALAR); // Create a scalar dataspace for the attributes
    
    hid_t dataset_id = H5Dopen(fileId, datasetName, H5P_DEFAULT);

    hid_t attribute_id = H5Acreate(
        dataset_id, 
        attributeName, 
        H5T_NATIVE_DOUBLE,
        attributespace_id, 
        H5P_DEFAULT, 
        H5P_DEFAULT
    );

    herr_t status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &attributeValue);
    
    H5Aclose(attribute_id);
    H5Sclose(attributespace_id);
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

        // Create and initialize a 1D array of 16 integers
        std::array<int, 16> arrayToWrite;

        for (size_t i = 0; i < arrayToWrite.size(); i++) {
            arrayToWrite[i] = (int)i;
        }

        // ### PART 1 ###
        hid_t fileId = H5Fopen(
            "my_datasets.h5",  // File name
            H5F_ACC_TRUNC , // Override existing file if it exists
            H5P_DEFAULT,     // Default file creation properties
        );

        // Write some datasets to the file.
        writeDataset1D(arrayToWrite, fileId, "MyDataset1D");
        writeDataset2D(arrayToWrite, fileId, "MyDataset2D");
        writeDataset3D(arrayToWrite, fileId, "MyDataset3D");

        // ### PART 2 ###
        // Write some metadata to the datasets
        writeAttribute("CoolAttribute", 42.0, fileId, "MyDataset1D");
        writeAttribute("CoolestAttribute", 0.577, fileId, "MyDataset3D");

        // Close the file
        H5Fclose(fileId);
    }

    MPI_Finalize();
    return 0;
}
