# transfromers version 4.36.2
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.starcoder2.modeling_starcoder2 import Starcoder2Attention
import json
from functools import partial
from modify_utils import modify_method_of_instance
import mistral_self_extend_patch as MistralSE
import warnings
warnings.filterwarnings("ignore")


original_sc2_forward = Starcoder2Attention.forward
self_extend_forward = partial(
    MistralSE.self_extend_forward, group_size_1=4, group_size_2=1024)


model_path = 'bigcode/starcoder2-3b'
config = transformers.AutoConfig.from_pretrained(model_path)
# disable mistral's default SWA mechanism (4096), mistral's true window is 8192.
config.sliding_window = 200000000
model = AutoModelForCausalLM.from_pretrained(
    model_path, config=config).to("cuda")


tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# In the example task file, the passkey is placed within the last 4096 tokens, this means, if you use SWA, mistral will successfully find the passkey.
for line in open("passkey_examples_10k.jsonl", "r"):
    example = json.loads(line)
    prompt_postfix = "What is the pass key? The pass key is "
    prompt = example["input"] + prompt_postfix
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids
    print("-----------------------------------")
    print(f"#Tokens of Prompt:", input_ids.shape[1], end=" ")
    print("Passkey target:", example["target"])

    modify_method_of_instance(
        model, "Starcoder2Attention", "forward", original_sc2_forward)
    tokens = model.generate(input_ids, max_new_tokens=6)
    answer = "StarCoder2:    [" + prompt_postfix + tokenizer.decode(
        tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True) + "]"
    answer = answer.replace("\n", "\\n")
    print(answer)

    modify_method_of_instance(
        model, "Starcoder2Attention", "forward", self_extend_forward)
    tokens = model.generate(input_ids, max_new_tokens=6)
    answer = "SelfExtend: [" + prompt_postfix + tokenizer.decode(
        tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True) + "]"
    answer = answer.replace("\n", "\\n")
    print(answer)
    print("-----------------------------------\n")
