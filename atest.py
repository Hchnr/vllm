# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams()

MODEL_PATH = "/share/project/hcr/models/wenxinyiyan/ernie34T-4l_torch_split"
MODEL_PATH = "/share/project/hcr/models/wenxinyiyan/ernie45T-trans"

def main():
    # Create an LLM.
    llm = LLM(model=MODEL_PATH, trust_remote_code=True, tensor_parallel_size=8, gpu_memory_utilization=0.96, max_model_len=40960)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    t_start = time.time()
    main()
    print(f"time cost: {time.time() - t_start}")
