## Example GPU

Consider the following example GPU.


![](../../../docs/img/gpu_as_cus_sms_eus.png)

![](../../../docs/img/cu_sm_eu.png)


Think about the following questions and try to come up with an answer
to them.

Consult the slides for help.

In the following, by "saturate the entire ..." we mean there's at least
a single unit of work for every piece of hardware.

For example, "to saturate an entire SIMD unit, four lanes wide" we need at least
four pieces of data.

----------------------------------------------------------------------

Base assumptions:
- the warp size is 32

While answering these, consider the example GPU and SM/CU given at the beginning.


### Grids, blocks, warps, threads

1. How many grids does it take, at least, to saturate the entire GPU?
    - What hardware component does a grid of blocks map to?  
    A grid of blocks maps to one GPU device.
    - How many of them are there?  
    There is one GPU.

2. How many blocks does it take, at least, to saturate the entire GPU?
    - What hardware component does a block of threads map to?  
    Blocks maps to CUs.
    - How many of them are there?  
    There are 8 CUs, requiring at least 8 blocks.

3. What is the minimum amount of threads in a block to saturate an entire CU/SM?
    - What is the warp size?  
    We have a warp size of 32.
    - What hardware component does a warp map to?  
    SIMD unit.
    - How many of these components are there in a single CU/SM?  
    There are 4 SIMD units.  
    Combining these facts, we require 32*4=128 threads.

4. How many threads does it take, in total, to saturate the entire GPU?
    - How many threads to sature a CU/SM (See question 3)?  
    - How many blocks to saturate the entire GPU (see question 2)?  
    We require 128*8=1024 threads.

### Miscellaneous

1. Can you, the user, control the number of threads in a warp (given a particular GPU)?  
No, it is vendor dependent (32 NVIDIA, 64 AMD).
2. Can you, the user, control the number of warps in a block?  
The number of warps depends on the number of threads.
3. Can you, the user, control the number of threads in a block?  
Yes, when defining the dimensions of the block, you specify the number of threads.  
4. If you answered 'No' to question 2, consider it again after answering question 3.
