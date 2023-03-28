import time
import torch
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity

def launch_kernels(large_matrices, small_matrices):

    # Block 1: execute large matmul followed by small matmuls
    torch.matmul(large_matrices[1], large_matrices[1])
    for j in range(10, 20):
       torch.matmul(small_matrices[j], small_matrices[j])

    torch.cuda.synchronize()
    time.sleep(delay)

    # Block 2: execute small matmuls followed by large matmul
    for i in range(10):
       torch.matmul(small_matrices[i], small_matrices[i])
    torch.matmul(large_matrices[0], large_matrices[0])


# set matrix size and device
large = 1600
small = 100
delay = 0.0005 # in seconds
cuda = torch.device('cuda')

# create small and large matrices
large_matrices = []
for _ in range(2):
    large_matrices.append(torch.rand((large, large), device='cuda'))

small_matrices = []
for _ in range(20):
    small_matrices.append(torch.rand((small, small), device='cuda'))

profile_steps = 10
for i in range(profile_steps):
    torch.cuda.cudart().cudaProfilerStart()
    # output = model(**encoded_input)
    launch_kernels(large_matrices, small_matrices)


torch.cuda.cudart().cudaProfilerStop()

#nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true -o my_profile python nsightsystem.py 
