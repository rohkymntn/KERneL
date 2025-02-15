import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

SAMPLE_NAME = "matmul_cuda"
SAMPLE_FUNCTION = "matrix_multiply_cuda"
with open("sample/kernel.cu", "r") as file:
    matmul_cuda_source = file.read()
with open("sample/kernel.cpp", "r") as file:
    matmul_cpp_source = file.read()

kernel_module = load_inline(
    name=SAMPLE_NAME,
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=[SAMPLE_FUNCTION],
    verbose=True
)

def time_execution_with_cuda_event():
    from sample.kernel import Model, get_inputs
    torch_model = Model()
    inputs = get_inputs()

    device = torch.cuda.current_device()
    elapsed_times_torch = []
    elapsed_times_kernel = []
    num_trials = 100

    for trial in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        torch_model(*inputs)
        end_event.record()

        torch.cuda.synchronize(device=device)
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print("Torch: ", elapsed_time_ms)
        elapsed_times_torch.append(elapsed_time_ms)

        start_event.record()
        getattr(kernel_module, SAMPLE_NAME)(*inputs)
        end_event.record()

        torch.cuda.synchronize(device=device)
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print("Kernel: ", elapsed_time_ms)
        elapsed_times_kernel.append(elapsed_time_ms)

    return elapsed_times_torch, elapsed_times_kernel

time_execution_with_cuda_event()