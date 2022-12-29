
# Why GPUs are fast?

![GPUs_flowchart](https://user-images.githubusercontent.com/9162336/209252228-0b1c74fc-e2d6-4ea6-9e59-ccf0dfba6122.png)

**Goal here is to maximize the useful computation**

GPU code issues usually are either **memory bound** or **compute bound**. We need to make sure we are using a good percentage of the memory banddwidth and at the same time optimize the compute bound issues as well.

For the first challenge lets look at the following for **memory bound** issues:

## Memory bound issues

### Measure and improve memory badnwidth
 * **Assure sufficient occupancy on GPU
 * **Coalece global memory access

#### Theoratical peak bandwidth

- `device query` run the command to get device info, important ones:
* memory clock: 2508 MHZ ==> 2508 x 10^6 clock/sec
* memory bus:128 bits ==> 16byte/ clock 
* max bandwidth = memory clock x memory bus = 40 GB/S
- if the DRAM utilizatin is 40-60% okay, 60-70% good, > 75% excellent

#### example of a Transpose kernel badnwidth

- suppose we have a 1024 x 1024 matrix
- total execution time = 0.6 ms
- peak bandwidth = (1024 x 1024 x 4(bytes) x 2(reading and writing transfer across the bus))/ 0.6(total time) = 1.25 10^10 = 12 GB
- can use profilers NVVP, NSIGHT, run Analyze all and can look at the DRAM untilization


**Paralallization would help to get higher bandwidth, techniques such as tiling, and others.**

### How to optimize

**Most GPU codes are memory limit**, 
1- measure the memory bandwidth, if low utilization then ask why?
- bad coalesing in read or write? could be a big hit on performance, tiling could help here
- <img width="1361" alt="Screen Shot 2022-12-22 at 5 11 17 PM" src="https://user-images.githubusercontent.com/9162336/209251396-b184c471-020f-4f01-a748-07a0c70a2068.png">


For the second challenge lets look at the following for **compute bound** issues:

## Compute bound issues

After analyzing and addressing memory bound issue as much as possible, we need to look at the computation part.

**Goal** here is to maximize the useful computation, if the code is compute bound, high level usually two main actions here we need take

* **minimize the time waiting at barriers**( this can happen when for example our tiling block is running a larger block size, laucnhing larger number of threads in the thread blocks)

* **minimize the thread divergence**
  * Warp: set of threads that execute same instruction at a time
  * SIMD: single instruction multiple data, you can amortize the work on fetching, decoding the instrcution to run on multiple data,( think of CPUs SSE or AVX vector instructions, where it will effect  4-8 pieces of data per instruction)
  * SIMT: single instruction multiple threads ( this matter when we have branch divergence)
    * each warp can take 32 threads
    * one example is if there are control flow in the code, some threads execute the if branch while others execute the else branch, as threads in a warp can execute a single instruciton at the time hardware automatically shutdown some of the threads resultsing in slow down of the code (thread divergence)
    * <img width="1382" alt="Screen Shot 2022-12-23 at 10 58 53 AM" src="https://user-images.githubusercontent.com/9162336/209394791-f392b20f-2253-4eb5-94c4-8a5492cbabe3.png">


## System level optimization ( considering CPU & GPU interaction)

### Pinned Memory
![device-host](https://user-images.githubusercontent.com/9162336/209401116-f101f341-058f-4ae1-83eb-ed6692809887.png)

- CPU (host) to GPU (device) interaction is using PCIe with max of 6GB/s data badnwidth
- Copy Host 2 Device, in CPU memory data frist will be copied to pinned memory then from there copied to GPU.
- using CUDA Host Malloc, will directly allocated memory on pinned memory on CPU so will skip the copy to pinned memory part that is time consuming.
-  Thats why in PyTorch pinning memory is an optimizaiton for dataloaders to run faster.
-  CUDA Memcpy Async, let the CPU work while copy to device is in progress.
-  <img width="600" alt="Screen Shot 2022-12-23 at 12 09 45 PM" src="https://user-images.githubusercontent.com/9162336/209401585-8a1c6c33-e146-4fe5-95e4-97c014ceae60.png">

### CUDA streams
Streams are a sequence of operations that execute in order (memory transfers, kernels): 
- Using multiple streams can parallelize and speed up the process
- it can overlap data transfer( communication) and computation
- Help to fill GPU with smaller kernels

- <img width="600" alt="Screen Shot 2022-12-23 at 12 09 45 PM" src="https://user-images.githubusercontent.com/9162336/209402432-4238be9b-b18e-4a08-b3f8-aa13754e59af.png">

### Patterns of Parallelization

1. Data layout transformation:
 * Memroy coalescing
 * Array of structured of tiled arrays (ASTA)
 
2. Scatter to gather tranasformation
 * Gather : many overlapping reads
 * Scatter : many potentially conlciting writes
 
3. Tiling : buffering data into on-chip fast memory for repeated access

4. Privatization : give each thread copy of the data to write in a specific memory location and finally do an all reduce

5. Binning : build a data structure that maps output data to the relevant (small set of )input data

6. Compaction: making a dense array from an sparse array and compute parallelization/multi-threading on them
7. Regularization : load balancing in distribution of work overs threads ( where mostly work is distributed evenly with some great outliers)
