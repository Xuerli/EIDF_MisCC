from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from sys import argv


if __name__ == "__main__":
    cache_dir = "/mnt/hf_cache"
    token = argv[1]
    dataset_file = "/mnt/toy_trees.json"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    output_file = "/mnt/predictions.jsonl"


    with open(dataset_file, "r") as f:
        instances = json.load(f)


    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token,
        cache_dir=cache_dir
    )

    messages = [{"role": "system", "content": "You are a system for content moderation. You analyse hateful and not hateful posts looking for elements that might lead to policy violations and explicitly tag them."}]

    # for i in range(10):
    #     messages.append({"role": "user", "content": instances[i]["input"]})
    #     messages.append({"role": "assistant", "content": instances[i]["output"]})

    generated_outputs = []

    for i in range(0, len(instances)):
        input_ids = tokenizer.apply_chat_template(
            messages + [{"role": "user", "content": instances[i]["input"]}],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            num_return_sequences=3,
        )

        response = outputs[0][input_ids.shape[-1]:]

        generated_outputs.append({
            "input": instances[i]["input"],
            # "output": instances[i]["output"],
            "generated_output": tokenizer.decode(response, skip_special_tokens=True),
            })

    with open(output_file, "w+") as out:
        for output in generated_outputs:
            json.dump(output, out)
            out.write("\n")
    out.close()
    print("Done!")
