#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // Set up queue on any available device
  queue q;
  
  // Initialize input and output memory on the host
  constexpr size_t N = 25600;
  //std::vector<int> x(N),y(N);
  int a=4;
  int *x = malloc_shared<int>(N, q);
  int *y = malloc_shared<int>(N, q);
  std::fill(x, x + N, 1);
  std::fill(y, y + N, 2);

   // Create buffers for the host data or allocate memory using USM
   // If USM + malloc_device() is used add the copy operations
    //int *d_x = malloc_device<int>(N, q);
    //int *d_y = malloc_device<int>(N, q);
    //buffer<int, 1> buf_x(x.data(), range<1>(N));
    //buffer<int, 1> buf_y(y.data(), range<1>(N));

    //q.memcpy(d_x, x.data(), N * sizeof(int)).wait();
    //q.memcpy(d_y, y.data(), N * sizeof(int)).wait();

    // Submit the kernel to the queue
    q.submit([&](handler& h) {
      // Create accessors if necessary
      //accessor x_acc(buf_x, h, read_only);
      //accessor y_acc(buf_y, h, read_write);

      //h.parallel_for(range<1>(N), [=](id<1> idx) {
      //  // The kernel code
      //  y_acc[idx] = a * x_acc[idx] + y_acc[idx];
      //});
      int local_size = 256; // Define local work-group size
      int global_size = ((N + local_size - 1) / local_size) * local_size; // Calculate global size

      h.parallel_for(nd_range<1>(range<1>(global_size), range<1>(local_size)), [=](nd_item<1> item) {
        auto idx=item.get_global_id(0);
        if(idx<N){ //to avoid out of bounds access
          //y_acc[idx] = a*x_acc[idx] + y_acc[idx];
          y[idx] = a * x[idx] + y[idx]; // Using USM pointers
        }
      });
    });

      //Checking the result inside the scope of the buffers using host_accessors
      {
          //host_accessor h_accY(buf_y, read_only); // Read back data after kernel execution
          //q.memcpy(y.data(), d_y, sizeof(int) * N).wait();
          std::cout << "First few elements of Y after operation:" << std::endl;
          for (size_t i = 0; i < 10; ++i) {
            std::cout << "Y[" << i << "] = " << y[i] << std::endl;
          }
      }
      
  // If USM + malloc_device() is used add the copy operations 
  // TODO
  // Check that all outputs match expected value

  // If USM is used free the device memory
  free(x, q);
  free(y, q);
  // Check that all outputs match expected value
  bool passed = std::all_of(y, y + N,
                            [a](int val) { return val == a * 1 + 2; });
  std::cout << ((passed) ? "SUCCESS" : "FAILURE")
            << std::endl;
  return (passed) ? 0 : 1;
}
