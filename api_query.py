from openai import OpenAI
import json

API_KEY = "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"
PROMPT_PREFIX_PATH = "prompt_prefix.txt"
PROMPT_POSTFIX_PATH = "prompt_postfix.txt"

# Initialize client
def initialize_client(api_key = API_KEY, base_url="https://integrate.api.nvidia.com/v1"):
    return OpenAI(
        base_url=base_url,
        api_key=api_key
    )

def query_kernel(client, model_type, pytorch_function, response_format="text", stream=True):
    with open(PROMPT_PREFIX_PATH, 'r') as file:
        prompt_prefix = file.read()
    with open(PROMPT_POSTFIX_PATH, 'r') as file:
        prompt_postfix = file.read()

    system_prompt = f"{prompt_prefix} {pytorch_function} {prompt_postfix}"

    completion = client.chat.completions.create(
        model=model_type,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write a CUDA kernel for the following PyTorch function:\n{pytorch_function}"}
        ],
        temperature=0.6,
        top_p=0.7,
        max_tokens=1000,
        stream=stream
        #response_format={'type': 'json_object'} if response_format == "json" else None,
    )

    return completion

def query_refine(response_text, refinement_client, model_type, stream = True):
    system_prompt = """
    You are given a text. Identify and extract the CUDA kernel code from the text. 
    Only return the CUDA kernel code. Do not return any additional explanation or comments.
    """

    completion = refinement_client.chat.completions.create(
        model=model_type,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract the CUDA kernel code from the following text:\n{response_text}"}
        ],
        temperature=0.6,
        top_p=0.7,
        max_tokens=100,
        stream=stream
    )
    
    return completion

def process_response(completion, refinement_client, refinement_type):
    print("Processing Response...")

    reasoning_response = []
    for chunk in completion:
        #print("Processing text chunk...")
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="")
        reasoning_response.append(content)
    print("\nFinished printing all")

    reasoning_response = "".join(reasoning_response)

    cuda_completion = query_refine(response_text = reasoning_response, 
                                   refinement_client = refinement_client, 
                                   model_type = refinement_type, 
                                   stream = True)

    cuda_response = []
    for chunk in cuda_completion:
        #print("Processing text chunk...")
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="")
        cuda_response.append(content)
    print("\nFinished printing all")

    return "".join(cuda_response)

if __name__ == '__main__':
    pytorch_function = """
    import torch

    # Example PyTorch function: element-wise addition
    def pytorch_addition(a, b):
        return a + b
    """

    # Initialize client and query API
    BASE_URL = "https://integrate.api.nvidia.com/v1"
    EXTRACTION_MODEL = "meta/llama-3.2-3b-instruct"#"tiiuae/falcon3-7b-instruct"
    DEEPSEEKR1_MODEL = "deepseek-ai/deepseek-r1"

    deepseek_client = initialize_client(api_key = API_KEY, base_url = BASE_URL)
    print("Client Initialized")
    completion = query_kernel(client = deepseek_client, 
                           model_type = DEEPSEEKR1_MODEL, 
                           pytorch_function = pytorch_function, 
                           response_format= "json", 
                           stream=True)
    print("Response Found")
    
    refinement_client = initialize_client(api_key = API_KEY, base_url = BASE_URL)
    process_response(completion, refinement_client, refinement_type = EXTRACTION_MODEL)

    print("All Done")


