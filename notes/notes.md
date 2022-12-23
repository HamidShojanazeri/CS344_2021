
# Why GPUs are fast?

![GPUs_flowchart](https://user-images.githubusercontent.com/9162336/209252228-0b1c74fc-e2d6-4ea6-9e59-ccf0dfba6122.png)

## Theoratical peak bandwidth

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


