import time
from fastdeploy import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams()

def test_all():
    MODEL_PATH = "/share/project/hcr/models/wenxinyiyan/ernie45T-trans"
    llm = LLM(model=MODEL_PATH, trust_remote_code=True, tensor_parallel_size=8, gpu_memory_utilization=0.96, max_model_len=40960)
    outputs = llm.generate(prompts, sampling_params)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs.text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)

def test_4l():
    MODEL_PATH = "/share/project/hcr/models/wenxinyiyan/ernie34T-4l"
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=1, gpu_memory_utilization=0.6, engine_worker_queue_port=9420)
    outputs = llm.generate(prompts, sampling_params)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs.text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    t_start = time.time()
    test_4l()
    print(f"time cost: {time.time() - t_start}")