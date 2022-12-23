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

- Paralallization would help to get higher bandwidth, techniques such as tiling, and others.
