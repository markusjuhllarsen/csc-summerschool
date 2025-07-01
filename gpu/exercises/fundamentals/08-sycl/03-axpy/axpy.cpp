#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // Set up queue on any available device
  queue q;
  
  // Initialize input and output memory on the host
  constexpr size_t N = 25600;
  std::vector<int> x(N),y(N);
  int a=4;
  std::fill(x.begin(), x.end(), 1);
  std::fill(y.begin(), y.end(), 2);

  {
   // Create buffers for the host data or allocate memory using USM
   // If USM + malloc_device() is used add the copy operations
    buffer<int, 1> buf_x(x.data(), range<1>(N));
    buffer<int, 1> buf_y(y.data(), range<1>(N));

    // Submit the kernel to the queue
    q.submit([&](handler& h) {
      // Create accessors if necessary
      auto x_acc = sycl::accessor{buf_x, h, sycl::read};
      auto y_acc = sycl::accessor{buf_y, h, sycl::read_write};

      h.parallel_for(range<1>(N), [=](id<1> idx) {
        // The kernel code
        y_acc[idx] = a * x_acc[idx];
      });
    });

      //Checking the result inside the scope of the buffers using host_accessors
      {
          host_accessor h_accY(buf_y, sycl::read); // Read back data after kernel execution
          std::cout << "First few elements of Y after operation:" << std::endl;
          for (size_t i = 0; i < 10; ++i) {
            std::cout << "Y[" << i << "] = " << h_accY[i] << std::endl;
          }
      }
  }
  // If USM + malloc_device() is used add the copy operations 
  // TODO
  // Check that all outputs match expected value

  // If USM is used free the device memory
  // TODO
  // Check that all outputs match expected value
  bool passed = std::all_of(y.begin(), y.end(),
                            [a](int val) { return val == a * 1 + 2; });
  std::cout << ((passed) ? "SUCCESS" : "FAILURE")
            << std::endl;
  return (passed) ? 0 : 1;
}
