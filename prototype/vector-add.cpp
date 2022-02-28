//==============================================================
// Vector Add is the equivalent of a Hello, World! sample for data parallel
// programs. Building and running the sample verifies that your development
// environment is setup correctly and demonstrates the use of the core features
// of DPC++. This sample runs on both CPU and GPU (or FPGA). When run, it
// computes on both the CPU and offload device, then compares results. If the
// code executes on both CPU and offload device, the device name and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding DPC++ Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// DPC++ material used in the code sample:
// •	A one dimensional array of data.
// •	A device queue, buffer, accessor, and kernel.
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#include "FakeIOPipes.hpp"
#include "HostSideChannel.hpp"


#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

struct DeviceToHostSideChannelID;
struct SideChannelMainKernel;
using MyDeviceToHostSideChannel = DeviceToHostSideChannel<DeviceToHostSideChannelID, int, true, 8>;
// Vector type and data size for this example.
size_t vector_size = 10000;
typedef std::vector<int> IntVector; 

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};



//************************************
// Vector add in DPC++ on device: returns sum in 4th parameter "sum_parallel".
//************************************
void VectorAdd(queue &q, const IntVector &a_vector, const IntVector &b_vector,
               IntVector &sum_parallel, IntVector &flag) {
  // Create the range object for the vectors managed by the buffer.
  range<1> num_items{a_vector.size()};
  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  buffer a_buf(a_vector);
  buffer b_buf(b_vector);
  buffer sum_buf(sum_parallel.data(), num_items);
  //buffer f_buf(flag.data(), num_items/2);
  MyDeviceToHostSideChannel::Init(q);
  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  q.submit([&](handler &h) {
    // Create an accessor for each buffer with access permission: read, write or
    // read/write. The accessor is a mean to access the memory in the buffer.
    accessor a(a_buf, h, read_only);
    accessor b(b_buf, h, read_only);

    // The sum_accessor is used to store (with write permission) the sum data.
    accessor sum(sum_buf, h, write_only, no_init);
    //accessor flag_kernel(f_buf, h, write_only, no_init);
    // Use parallel_for to run vector addition in parallel on device. This
    // executes the kernel.
    //    1st parameter is the number of work items.
    //    2nd parameter is the kernel, a lambda that specifies what to do per
    //    work item. The parameter of the lambda is the work item id.
    // DPC++ supports unnamed lambda kernel by default.
    h.parallel_for(num_items, [=](auto i) { sum[i] = a[i] + b[i]; 
                                            if (sum[i]<0){MyDeviceToHostSideChannel::write(i);}
                                        
                                          });
  });
  for (int i = 0; i < 3; i++) {
        // Blocking read an int from the pipe
        flag[i] = MyDeviceToHostSideChannel::read();
      }
  
}

int write_file(const char *address, const IntVector &a, const IntVector &b){
    FILE* f = fopen(address,"w");
    
    int aa=0;
    for (int i = 0; i < a.size(); i++) {
      aa  = a[i];
      fprintf(f,"%d", aa);
    }
    for (int i = 0; i < b.size(); k++) {
      aa = b[i];
      fprintf(f,"%d", aa);
    }
    return 0;
}


//************************************
// Initialize the vector from 0 to vector_size - 1
//************************************
void InitializeVector(IntVector &a) {
  for (size_t i = 0; i < a.size(); i++) a.at(i) = i;
  a.at(2) = (1<<30) + 2000000;
}


void mutate(IntVector &a, IntVector &b, int max_i){
    srand(time(NULL) + rand());
    int knob = rand()%4+1;
    std::thread *threads = new std::thread[THREADS];

    int pos_i = rand()% max_i;
    
    if (knob==1){
        float value = (double)(rand()) / ((double)(RAND_MAX/MAX));
        a[pos_i] = value;
    }
    else if (knob==2){
        a[pos_i] = 0;
    }
    else if (knob==3){
        int max_k = rand()% (max_i - pos_i)
        for (size_t k = pos_i; k < pos_i+max_k; k++){
            float value = (double)(rand()) / ((double)(RAND_MAX/MAX));
            a[k] = value;
        }
    }
    else if (knob==4){
        for (int i = 0; i <pos_i; i++){
            a[i] = 0;
        }
        //for (int i = 0; i < THREADS; i++)
		//    threads[i] = std::thread(parallel_sparsity, std::ref(a), max_k, max_i, i);
	    //for (int i = 0; i < THREADS; i++)
		//    threads[i].join();
    }
    else if (knob==5){
        for (int i = 0; i < pos_i; i++){
            a[i] = a[i] + 100;
        }
        //for (int i = 0; i < THREADS; i++)
		//    threads[i] = std::thread(parallel_add, std::ref(a), max_k, max_i, i);
	    //for (int i = 0; i < THREADS; i++)
		//    threads[i].join();
    }
    else if (knob==6){
        for (int i = 0; i < pos_i; i++){
            a[i] = a[i] - 100;
        }
        //for (int i = 0; i < THREADS; i++)
		//    threads[i] = std::thread(parallel_minus,std::ref(a), max_k, max_i, i);
	    //for (int i = 0; i < THREADS; i++)
		//    threads[i].join();
    }
}
//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char* argv[]) {
  // Change vector_size if it was passed as argument
  if (argc > 1) vector_size = std::stoi(argv[1]);
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  ext::intel::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  default_selector d_selector;
#endif

  // Create vector objects with "vector_size" to store the input and output data.
  IntVector a, b, sum_sequential, flag, sum_parallel;
  a.resize(vector_size);
  b.resize(vector_size);
  flag.resize(vector_size/2);
  sum_sequential.resize(vector_size);
  sum_parallel.resize(vector_size);

  // Initialize input vectors with values from 0 to vector_size - 1
  InitializeVector(a);
  InitializeVector(b);

  int file_number = 0;
  int current_file = 0;


  try {
    queue q(d_selector, exception_handler);

    for (int i = 0; i<6; i++){
    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Vector size: " << a.size() << "\n";

    // Vector addition in DPC++

    int interest = VectorAdd(q, a, b, sum_parallel, flag);
    mutate(a, vector_size);
    mutate(b, vector_size);
    if (interest==1) {
        std::string path_to_output(argv[1]);
        write_file((path_to_output+"-"+to_string(file_number)).c_str(),a,b);
    }
    
    }
  } catch (exception const &e) {
    std::cout << "An exception is caught for vector add.\n";
    std::terminate();
  }

  // Compute the sum of two vectors in sequential for validation.
  for (size_t i = 0; i < sum_sequential.size(); i++)
    sum_sequential.at(i) = a.at(i) + b.at(i);

  // Verify that the two vectors are equal.  
  for (size_t i = 0; i < sum_sequential.size(); i++) {
    if (sum_parallel.at(i) != sum_sequential.at(i)) {
      std::cout << "Vector add failed on device.\n";
      return -1;
    }
  }

  int indices[]{0, 1, 2, (static_cast<int>(a.size()) - 1)};
  constexpr size_t indices_size = sizeof(indices) / sizeof(int);

  // Print out the result of vector add.
  for (int i = 0; i < indices_size; i++) {
    int j = indices[i];
    if (i == indices_size - 1) std::cout << "...\n";
    std::cout << "[" << j << "]: " << a[j] << " + " << b[j] << " = "
      << sum_parallel[j] << " " << flag[i/2]<<"\n";
  }

  a.clear();
  b.clear();
  sum_sequential.clear();
  sum_parallel.clear();

  std::cout << "Vector add successfully completed on device.\n";
  return 0;
}