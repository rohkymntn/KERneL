import sys
import torch
import torch.nn as nn
import importlib.util
from uuid import uuid4
from flask import Flask, request, jsonify
from torch.utils.cpp_extension import load_inline
from llm_query_sample import llm_query

TASKS = {}


def initialize_python_module(source, module_name="dynamic_kernel_module"):
    try:
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        dynamic_module = importlib.util.module_from_spec(spec)
        exec(source, dynamic_module.__dict__)
        sys.modules[module_name] = dynamic_module
        return True, dynamic_module
    except Exception as e:
        return False, str(e)


def initialize_kernel_module(cuda_sources, cpp_sources, function_name, kernel_name="dynamic_cuda_kernel"):
    try:
        kernel_module = load_inline(
            name=kernel_name,
            cuda_sources=cuda_sources,
            cpp_sources=cpp_sources,
            functions=[function_name],
            verbose=True
        )

        return True, kernel_module
    except Exception as e:
        return False, str(e)


def time_execution_with_cuda_event(model, inputs, num_trials):
    device = torch.cuda.current_device()
    model.to(device=device)
    inputs = [inp.to(device=device) for inp in inputs]

    elapsed_times = 0
    for _ in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        model(*inputs)
        end_event.record()

        torch.cuda.synchronize(device=device)
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_times += elapsed_time_ms

    return elapsed_times / num_trials


app = Flask(__name__)

@app.route('/initialize_task', methods=['POST'])
def initialize_task():
    try:
        data = request.get_json()
        python_source = data.get("python_source", "")

        initialized, result = initialize_python_module(python_source)
        if initialized:
            Model, get_inputs = getattr(result, "Model"), getattr(result, "get_inputs")

            model = Model()
            inputs = get_inputs()

            task_id = uuid4()
            TASKS[task_id] = [model, inputs]

            average_time = time_execution_with_cuda_event(model, inputs)
            response = {
                "status": "Task initialized successfully",
                "torch_time": average_time,
                "task_id": task_id
            }
            return jsonify(response), 200
    except Exception as e:
        response = {"error": str(e)}
        return jsonify(response), 400


@app.route('/get_kernel', methods=['POST'])
def get_kernel():
    try:
        data = request.get_json()

        task_id = data.get("task_id", "")
        _, inputs = TASKS[task_id]

        num_trials = data.get("num_trials", 100)
        function_name, cuda_sources, cpp_sources = llm_query()
        initialized, result = initialize_kernel_module(cuda_sources, cpp_sources, function_name)

        if initialized:
            model = getattr(result, function_name)
            average_time = time_execution_with_cuda_event(model, inputs, num_trials)

            response = {
                "task_id": task_id,
                "status": "Kernel compiled successfully",
                "kernel_time": average_time
            }
            return jsonify(response), 200
        else:
            response = {
                "task_id": task_id,
                "status": "Failed during kernel compilation",
                "error": result
            }
            return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "task_id": task_id,
            "error": str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True)
