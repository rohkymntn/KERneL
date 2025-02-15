from openai import OpenAI
import json
from typing import Union, Tuple
import os
import requests
from datetime import datetime

os.environ["NIM_ENABLE_KV_CACHE_REUSE"] = "1"

API_KEY = "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"
PROMPT_PREFIX_PATH = "prompt_prefix.txt"
PROMPT_POSTFIX_PATH = "prompt_postfix.txt"
MAX_REASONING_TOKENS = 2000
MAX_REFINEMENT_TOKENS = 500

BASE_URL = "https://integrate.api.nvidia.com/v1"
EXTRACTION_MODEL = "qwen/qwen2.5-7b-instruct" #"meta/llama-3.2-3b-instruct"
DEEPSEEKR1_MODEL = "deepseek-ai/deepseek-r1"
  
def save_kernels(cpp_kernel, cuda_kernel, directory="sample"):
    # Ensure the output directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the current date and time for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the filenames
    cpp_filename = os.path.join(directory, f"kernel_{timestamp}.cpp")
    cu_filename = os.path.join(directory, f"kernel_{timestamp}.cu")

    # Save the cpp_kernel to a file
    with open(cpp_filename, "w") as cpp_file:
        cpp_file.write(cpp_kernel)
    print(f"Saved C++ kernel to {cpp_filename}")

    # Save the cuda_kernel to a file
    with open(cu_filename, "w") as cu_file:
        cu_file.write(cuda_kernel)
    print(f"Saved CUDA kernel to {cu_filename}")

def initialize_client(api_key = API_KEY, base_url="https://integrate.api.nvidia.com/v1"):
    return OpenAI(
        base_url = base_url,
        api_key = api_key
    )

def query_kernel(client: OpenAI, 
                 model_type: str, 
                 pytorch_function: str, 
                 additional_context: str = "",
                 stream = True) -> str:
    
    with open(PROMPT_PREFIX_PATH, 'r') as file:
        prompt_prefix = file.read()
    with open(PROMPT_POSTFIX_PATH, 'r') as file:
        prompt_postfix = file.read()

    if len(additional_context) == 0:
        system_prompt = f"{prompt_prefix} {pytorch_function} {prompt_postfix}"
    else:
        system_prompt = f"{prompt_prefix} {pytorch_function} Here is Feedback for Your Last Function {additional_context}  {prompt_postfix}"
    
    completion = client.chat.completions.create(
        model=model_type,
        messages= [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write a CUDA kernel for the following PyTorch function:\n{pytorch_function}"}
        ],
        temperature=0.6,
        top_p=0.7,
        max_tokens=MAX_REASONING_TOKENS,
        stream=stream
        #response_format={'type': 'json_object'} if response_format == "json" else None,
    )

    reasoning_response = []
    for chunk in completion:
        #print("Processing text chunk...")
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="")
        reasoning_response.append(content)
    
    print("\nFinished printing original response")

    return reasoning_response

def query_refine(response_text, refinement_client, model_type, system_prompt=None, stream=True):
    if system_prompt is None:
        system_prompt = """
        You are given a text that contains CUDA wrapper code for kernel.cpp and kernel.cu. 
        Your task is to extract the relevant code and return it structured within XML-like tags, as shown below:


        <kernel_cu>
        <insert the complete kernel.cu code here>
        </kernel_cu>


        <cpp_kernel>
        <insert the complete kernel.cpp code here>
        </cpp_kernel>


        Rules:
        1. Do not include any text or comments outside of the specified tags.
        2. Ensure that each section is complete and valid, containing only the respective code for kernel.cpp and kernel.cu.
        4. The output should end cleanly, with no extra tokens, trailing characters, or explanations.
        5. Ensure all quotes and special characters within the code are properly escaped.

        Return only the structured output as described above. Do not include any additional text, comments, or explanations.
        """
       
    completion = refinement_client.chat.completions.create(
        model=model_type,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract the CUDA and C++ wrapper code (kernel.cu and kernel.cpp) from the following text:\n{response_text}"}
        ],
        temperature=0.6,
        top_p=0.7,
        max_tokens= MAX_REFINEMENT_TOKENS,  # Adjusted for larger responses
        stream=stream
    )

    return completion

def extract_kernels(tagged_response):
    # Find the <cpp_kernel> section
    cpp_start = tagged_response.find("<cpp_kernel>")
    cpp_end = tagged_response.find("</cpp_kernel>") + len("</cpp_kernel>")
    cpp_kernel = tagged_response[cpp_start:cpp_end].strip() if cpp_start != -1 else "<cpp_kernel></cpp_kernel>"

    # Find the <kernel_cu> section
    cu_start = tagged_response.find("<kernel_cu>")
    cu_end = tagged_response.find("</kernel_cu>") + len("</kernel_cu>")
    kernel_cu = tagged_response[cu_start:cu_end].strip() if cu_start != -1 else "<kernel_cu></kernel_cu>"

    return cpp_kernel, kernel_cu

def process_response(reasoning_text: str, 
                     refinement_client: OpenAI, 
                     refinement_type: str) -> Tuple[str, str]:
    
    extracted_answers = query_refine(response_text = reasoning_text, 
                                   refinement_client = refinement_client, 
                                   model_type = refinement_type, 
                                   stream = True)

    cuda_response = []
    for chunk in extracted_answers:
        content = chunk.choices[0].delta.content
        if content:
            cuda_response.append(content)

    # Combine the response chunks into a full response string
    full_response = "".join(cuda_response).strip()

    print("\nFinished streaming response:")
    print(full_response)

    # Extract kernel.cpp and kernel.cu using XML-like tags
    try:
        # Find the content between <kernel_cu> and </kernel_cu>
        cu_start = full_response.find("<kernel_cu>") + len("<kernel_cu>")
        cu_end = full_response.find("</kernel_cu>")
        kernel_cu = full_response[cu_start:cu_end].strip() if cu_start != -1 and cu_end != -1 else ""

        # Find the content between <cpp_kernel> and </cpp_kernel>
        cpp_start = full_response.find("<cpp_kernel>") + len("<cpp_kernel>")
        cpp_end = full_response.find("</cpp_kernel>")
        cpp_kernel = full_response[cpp_start:cpp_end].strip() if cpp_start != -1 and cpp_end != -1 else ""

        # Return the extracted kernel.cpp and kernel.cu
        return cpp_kernel, kernel_cu
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None, None


def query_cuda_and_cpp_kernel(pytorch_function: str) -> Tuple[str, str]:
    # Initialize client and query API
    
    client = initialize_client(api_key = API_KEY, base_url = BASE_URL)
    print("Client Initialized")
    
    completion = query_kernel(client = client, 
                        model_type = DEEPSEEKR1_MODEL, 
                        pytorch_function = pytorch_function, 
                        response_format= "json", 
                        stream=True) 
    
    print("Response Found")

    cpp_kernel, cuda_kernel = process_response(completion, 
                     client, 
                     refinement_type = EXTRACTION_MODEL)

    return cpp_kernel, cuda_kernel

if __name__ == '__main__':
    pytorch_function = """
    import torch

    # Example PyTorch function: element-wise addition
    def pytorch_addition(a, b):
        return a + b
    """

    cpp_kernel, cuda_kernel = query_cuda_and_cpp_kernel(pytorch_function)

    save_kernels(cpp_kernel, cuda_kernel, directory = "sample")


def debug():

    deepseek_output = """
    <think>
    Okay, I need to create a CUDA kernel to replace the PyTorch element-wise addition function. Let me start by understanding what the original function does. It takes two tensors a and b and returns their element-wise sum. The goal is to write a custom CUDA kernel that performs this operation more efficiently.

    First, I'll outline the steps required. The CUDA kernel should take two input tensors, add each corresponding element, and store the result in an output tensor. I'll need to write the kernel function itself, the launch parameters (block size, grid size), and the PyTorch wrapper to integrate it.

    I remember that in CUDA, each thread handles one element or a chunk of elements. For simplicity, I'll start with one thread per element. The block size can be 256 threads, which is a common choice. The grid size will then be the total number of elements divided by the block size, rounded up.

    Next, I need to handle the kernel launch. The inputs are PyTorch tensors, so I have to access their data pointers. Also, I need to ensure that the tensors are on the GPU and contiguous. The output tensor should be initialized with zeros_like(a) to match the shape and device.

    Wait, what about different tensor shapes? The original PyTorch function broadcasts if necessary, but for simplicity, maybe I'll assume that a and b are the same shape. The user's example uses a and b of the same shape, so perhaps that's acceptable here.

    Now, writing the CUDA kernel. The kernel will have parameters for the input pointers, output pointer, and the total number of elements. Each thread calculates its global index and checks if it's within bounds before performing the addition.

    Then, the PyTorch extension. Using torch.utils.cpp_extension.load_inline to compile the CUDA code. The cpp_sources and cuda_sources need to be properly formatted. The function name in the C++ code should be exposed so that Python can call it.

    Testing the code: I need to make sure that the ModelNew class uses the custom CUDA kernel instead of the default addition. The forward method will call the compiled function. Also, the get_inputs function should generate tensors on the GPU.

    Potential issues: Forgetting to check the tensor dimensions, incorrect grid/block calculations, or memory access issues. Also, ensuring that the CUDA code is correctly formatted with proper includes and syntax.

    Let me put it all together. The CUDA kernel is defined in a string, then loaded as a PyTorch extension. The ModelNew class uses this extension in its forward pass. The input tensors are created on the GPU to match the kernel's expectations.

    I should also verify that the code compiles and runs without errors. Using load_inline with verbose=True might help catch any compilation issues. The block and grid dimensions are calculated correctly to cover all elements.

    Finally, the get_inputs function in the example uses tensors of size 4096, which is a good test size. The user's original example had 1x128 tensors, but larger sizes would better demonstrate the benefits of parallelization.
    </think>

    ```python
    import torch
    from torch.utils.cpp_extension import load_inline

    # Custom CUDA kernel implementation for element-wise addition
    cuda_source = '''
    #include <torch/extension.h>
    #include <cuda.h>
    #include <cuda_runtime.h>

    __global__ void elementwise_add_kernel(
        const float* a,
        const float* b,
        float* out,
        int num_elements
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_elements) {
            out[idx] = a[idx] + b[idx];
        }
    }

    torch::Tensor custom_add_cuda(torch::Tensor a, torch::Tensor b) {
        // Validate inputs
        TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be on GPU");
        TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be on GPU");
        TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
        
        auto out = torch::zeros_like(a);
        const int num_elements = a.numel();
        
        // Kernel launch parameters
        const int threads_per_block = 256;
        const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
        
        // Ensure tensors are contiguous
        auto a_cont = a.contiguous();
        auto b_cont = b.contiguous();
        
        elementwise_add_kernel<<<blocks, threads_per_block>>>(
            a_cont.data_ptr<float>(),
            b_cont.data_ptr<float>(),
            out.data_ptr<float>(),
            num_elements
        );
        
        return out;
    }
    '''

    cpp_source = "torch::Tensor custom_add_cuda(torch::Tensor a, torch::Tensor b);"

    # Load the custom CUDA extension
    custom_add = load_inline(
        name='custom_add',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['custom_add_cuda'],
        verbose=True
    )

    class ModelNew(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, a, b):
            return custom_add.custom_add_cuda(a, b)

    def get_inputs():
        a = torch.randn(4096, device='cuda')  # Larger tensor for meaningful GPU utilization
        b = torch.randn(4096, device='cuda')
        return [a, b]

    def get_init_inputs():
        return []
    ```

    Key optimizations and features:
    1. **Batched element-wise operations**: Processes 4096 elements for better GPU utilization
    2. **Optimal thread configuration**: Uses 256 threads per block (common best practice)
    3. **Memory coalescing**: Ensures contiguous memory access patterns
    4. **Input validation**: Checks tensor devices and shapes match
    5. **Asynchronous execution**: Leverages CUDA's parallel execution model
    6. **Zero-copy tensor handling**: Uses PyTorch's memory management directly

    To use:
    ```python
    model = ModelNew().cuda()
    a, b = get_inputs()
    result = model(a, b)
    ```

    This implementation should provide significant speedups over the native PyTorch implementation for large tensors due to parallel execution on GPU cores
    """


    if False:
        deepseek_client = initialize_client(api_key = API_KEY, base_url = BASE_URL)
        print("Client Initialized")
        completion = query_kernel(client = deepseek_client, 
                            model_type = DEEPSEEKR1_MODEL, 
                            pytorch_function = pytorch_function, 
                            response_format= "json", 
                            stream=True) 
        print("Response Found")
    else:
        completion = deepseek_output