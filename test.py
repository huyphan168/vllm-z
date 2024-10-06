from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.control_vectors.request import ControlVectorRequest
from transformers import AutoTokenizer

MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
cv_path = "/datadrive5/huypn16/ICV/controlvector.gguf"


def do_sample(engine):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt_text = "Solving the following mathematical problem. Problem: Calculate the following expression: (10222 + 23123123 * 4 - 1 ) * 5 - 6. Step 1:" 
    prompt = tokenizer(prompt_text, return_tensors="pt")
    print(len(prompt[0]))
    
    # first prompt with a control vector and second without.
    prompts = [(prompt_text,
                SamplingParams(temperature=0.65,
                               max_tokens=100),
                ControlVectorRequest("chaotic", 1, cv_path, scale=1.0)),
               (prompt_text,
                SamplingParams(temperature=0.0,
                               max_tokens=100), None)]

    request_id = 0
    results = set()
    while prompts or engine.has_unfinished_requests():
        if prompts:
            prompt, sampling_params, cv_request = prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               control_vector_request=cv_request)
            request_id += 1

        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.finished or request_output.outputs[0].prompt_hidden_states != None:
                results.add((request_output.request_id, request_output.outputs[0].text, request_output.outputs[0].hidden_states.shape, request_output.outputs[0].prompt_hidden_states.shape if request_output.outputs[0].prompt_hidden_states != None else None))
        if len(results) == 4:
            break
    return results


def test_cv_adapter():
    engine_args = EngineArgs(model=MODEL_PATH, enable_control_vector=True, gpu_memory_utilization=0.95, return_hidden_states=True)
    engine = LLMEngine.from_engine_args(engine_args)
    result = do_sample(engine)
    print(list(result))
    print(len(result))

if __name__ == "__main__":
    test_cv_adapter()