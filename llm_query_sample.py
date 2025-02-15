def llm_query(python_source, context):
    function_name = "diag_matmul_cuda"
    cuda = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(
    const float* diag,
    const float* mat,
    float* out,
    const int N,
    const int M) {
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < M) {
        out[row * M + col] = diag[row] * mat[row * M + col];
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor diag, torch::Tensor mat) {
    const int N = diag.size(0);
    const int M = mat.size(1);
    
    auto out = torch::zeros({N, M}, mat.options());
    
    const dim3 threads(16, 16);
    const dim3 blocks((M + threads.x - 1) / threads.x,
                     (N + threads.y - 1) / threads.y);
                     
    diag_matmul_kernel<<<blocks, threads>>>(
        diag.data_ptr<float>(),
        mat.data_ptr<float>(),
        out.data_ptr<float>(),
        N, M);
        
    return out;
}
"""
    cpp = "torch::Tensor diag_matmul_cuda(torch::Tensor diag, torch::Tensor mat);"

    return function_name, cuda, cpp