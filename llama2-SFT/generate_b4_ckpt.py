import os
import sys
import json
import argparse
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",
    saveI: int =0,
    from_json: str="",
    save_path: str=""
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=25,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)

    split_idx=4000
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    begin_idx=(saveI-1)*split_idx
    if os.path.exists(save_path+'/Answer'+str(saveI)+'.txt'):
        f=open(save_path+'/Answer'+str(saveI)+'.txt','r',encoding="utf8")
        data=f.read().split('##@#')
        begin_idx=(saveI-1)*split_idx+len(data)-1
    else:
        begin_idx=(saveI-1)*split_idx
        f=open(save_path+'/Answer'+str(saveI)+'.txt','w',encoding="utf8")
        f.write('')
        f.close()

    instructions=[]
    inputs=[]
    f=open(str(from_json),'r')
    data_f=json.load(f)
    for line in data_f:
        instructions.append(line['instruction'])
        inputs.append(line['input'])
    f.close()

    for instruction_idx in range(begin_idx,saveI*split_idx,1):
        instruction=instructions[instruction_idx]
        input_each=inputs[instruction_idx]
        responseL=list(evaluate(instruction=instruction,input=input_each))
        response=''
        for items in responseL:
            response+=items
        print("Response:", response)
        f=open(save_path+'/Answer'+str(saveI)+'.txt','a',encoding="utf8")
        f.write(str(instruction_idx)+"@@"+response+'\n##@#\n')
        f.close()

    '''
    while True:
        instruction = input("Input:")
        if len(instruction.strip()) == 0:
            break
        print("Response:", evaluate(instruction))
    '''


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_weights', default=None, type=str,
                        help="If None, perform inference on the base model")
    parser.add_argument('--load_8bit', action='store_true',
                        help='only use CPU for inference')
    parser.add_argument('--saveI', default=None, type=int, required=True)
    parser.add_argument('--from_json', default=None, type=str,
                        help="If None, perform inference on the base model")
    parser.add_argument('--save_path', default=None, type=str,
                        help="If None, perform inference on the base model")
    args = parser.parse_args()
    main(args.load_8bit, args.base_model, args.lora_weights, "",args.saveI,args.from_json,args.save_path)
