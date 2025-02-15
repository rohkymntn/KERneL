from openai import OpenAI
import json

API_KEY = "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"

# Initialize the DeepSeekR1 client
def initialize_client(api_key = API_KEY, base_url="https://integrate.api.nvidia.com/v1"):
    return OpenAI(
        base_url=base_url,
        api_key=api_key
    )

def query_api(client, model_type, pytorch_function, response_format="text", stream=True):
    if response_format == "json":
        system_prompt = """
        You are a CUDA kernel generator. Convert the provided PyTorch function into CUDA code.
        Ensure your response is a valid JSON object with the following structure:
        {
            "cuda_kernel": "<generated CUDA kernel>",
            "reasoning": "<explanation of the kernel logic>"
        }
        Always use proper JSON syntax with double quotes for property names and string values.
        """
    else:
        system_prompt = "You are a CUDA kernel generator."

    completion = client.chat.completions.create(
        model=model_type,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write a CUDA kernel for the following PyTorch function:\n{pytorch_function}"}
        ],
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096,
        response_format={'type': 'json_object'} if response_format == "json" else None,
        stream=stream
    )

    return completion

def process_response(completion, response_format="text"):
    print([chunk for chunk in completion])
    if response_format == "json":
        try:
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    result = json.loads(content)
                    print("Generated CUDA Kernel:")
                    print(result.get("cuda_kernel", "No CUDA kernel found"))
                    print("\nReasoning:")
                    print(result.get("reasoning", "No reasoning provided"))
                    return result
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
    else:
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        print("\nFinished printing all")

if __name__ == '__main__':
    pytorch_function = """
    import torch

    # Example PyTorch function: element-wise addition
    def pytorch_addition(a, b):
        return a + b
    """

    # Initialize client and query API
    BASE_FALCON3_URL = "https://integrate.api.nvidia.com/v1"
    FALCON3_MODEL = "tiiuae/falcon3-7b-instruct"

    client = initialize_client(api_key = API_KEY, base_url = BASE_FALCON3_URL)
    print("Client Initialized")
    completion = query_api(client = client, 
                           model_type = FALCON3_MODEL, 
                           pytorch_function = pytorch_function, 
                           response_format="json", 
                           stream=True)
    print("Response Found")
    
    # Process the response
    print("Completition Parsed")
    process_response(completion, response_format="json")
    print("All Done")


