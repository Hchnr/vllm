# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import torch
from vllm import LLM, SamplingParams

torch.set_printoptions(precision=8)

# Sample prompts.
prompts = [
    "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(seed=0, temperature=0.0001, top_p=0.8, max_tokens=1)

def test_all():
    MODEL_PATH = "/share/project/hcr/models/wenxinyiyan/ernie45T-trans"
    llm = LLM(model=MODEL_PATH, trust_remote_code=True, tensor_parallel_size=8, gpu_memory_utilization=0.95, max_model_len=1024, enforce_eager=True)
    sampling_params = SamplingParams(seed=0, temperature=0.0001, top_p=0.8, max_tokens=1)

    input_token_ids = [100273, 2969, 93963, 93919, 16276, 93938, 851, 853, 357, 23, 92267, 93963, 93919]
    outputs2 = llm.generate(prompt_token_ids=[input_token_ids], sampling_params=sampling_params)

    import pdb; pdb.set_trace()

    for output in outputs2:
        print("Generated text:", output.outputs[0].text)

    outputs = llm.generate(prompts, sampling_params)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


def test_4l():
    MODEL_PATH = "/share/project/hcr/models/wenxinyiyan/ernie34T-4l_torch_split"
    llm = LLM(model=MODEL_PATH, trust_remote_code=True, tensor_parallel_size=2, gpu_memory_utilization=0.8, max_model_len=40960, enforce_eager=True)
    '''
    outputs = llm.generate(prompts, sampling_params)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)
    '''

    input_token_ids = [100273, 2969, 93963, 93919, 16276, 93938, 851, 853, 357, 23, 92267, 93963, 93919]
    outputs2 = llm.generate(prompt_token_ids=[input_token_ids], sampling_params=sampling_params)
    for output in outputs2:
        print("Generated text:", output.outputs[0].text)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    t_start = time.time()
    test_4l()
    # test_all()
    print(f"time cost: {time.time() - t_start}")
