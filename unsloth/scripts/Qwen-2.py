import os

hf_token = os.environ.get('HF_TOKEN')
hf_token = None if hf_token == None else hf_token

org_model_name = os.environ.get('model_name')
max_seq_length=1024
#
# 加载模型
#
from unsloth import FastLanguageModel
import torch
print('load model : ' + org_model_name)

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit",  # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit",  # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit",  # [NEW] 15 Trillion token Llama-3
]  # More models at https://huggingface.co/unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=org_model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
    token=hf_token,
    device_map     = "sequential",
    rope_scaling   = None, # Qwen2 does not support RoPE scaling
    fix_tokenizer  = True,
    trust_remote_code = False
)

#
# 添加 LoRA 适配器, 仅需要 %10 的参数
#
print('add lora ...')
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    # lora_rank= 64,
    lora_alpha= 16,
    lora_dropout= 0.05,
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=True,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

#
# 准备数据集
#
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts, }


def formatting_prompts_func2(lines):
    texts = []
    for examples in lines['conversations']:
        q = examples[0]
        a = examples[1]
        inputs = ''
        text = alpaca_prompt.format(q['value'], inputs, a['value']) + EOS_TOKEN
        texts.append(text)

    return {"text": texts, }


pass

from datasets import load_dataset, load_from_disk

# 是否加载训练文件并转换为map
disk_hf = "/datasets/hf"
disk_train = "/datasets/train"
if len(os.listdir(disk_hf)) > 0:
    print('load disk hf')
    dataset = load_from_disk(disk_hf);
else:
    print('load disk train')
    dataset = load_dataset(disk_train, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True, )
    dataset.save_to_disk(disk_hf)

#
# 开始训练
#
print('SFTTrainer ...');
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        # per_device_train_batch_size=2,
        # gradient_accumulation_steps=4,
        # warmup_steps=5,
        # max_steps=60, #60
        # learning_rate=2e-4,
        # fp16 = not is_bfloat16_supported(),
        # bf16 = is_bfloat16_supported(),
        # logging_steps=1,
        # optim="paged_adamw_32bit", #Qwen paged_adamw_32bit
        # weight_decay=0.01,
        # lr_scheduler_type="linear",
        learning_rate= 2e-4,
        optim= "paged_adamw_32bit",
        lr_scheduler_type= "constant_with_warmup",
        warmup_steps= 5,
        max_steps= 1000,
        # num_train_epochs=200,
        # lora_rank= 64,
        # lora_alpha= 16,
        # lora_dropout= 0.05,
        # gradient_checkpointing= True,
        fp16= True,
        seed=3407,
        output_dir="outputs",
    ),
)

#
# 验证推理
#
def has_checkpoint(directory):
    files = os.listdir(directory)
    checkpoint_files = [file for file in files if file.startswith('checkpoint-')]
    return len(checkpoint_files) > 0
checkpoint = has_checkpoint('/outputs')
print('train - resume_from_checkpoint : ' + str(checkpoint))
trainer_stats = trainer.train(resume_from_checkpoint = checkpoint ) #检查点开始训练
print(trainer_stats)

#
# 推理模型
#
# print('train ...')
# FastLanguageModel.for_inference(model) # Enable native 2x faster inference
# inputs = tokenizer(
# [
    # alpaca_prompt.format(
        # "Continue the fibonnaci sequence.", # instruction
        # "1, 1, 2, 3, 5, 8", # input
        # "", # output - leave this blank for generation!
    # )
# ], return_tensors = "pt").to("cuda")

# outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
# tokenizer.batch_decode(outputs)

#
# 保存模型
#
print('save model ...')
model.save_pretrained("/outputs/model/lora_model")  # Local saving
tokenizer.save_pretrained("/outputs/tokenizer/lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

#
# Saving to float16 for VLLM
#
print('Saving to float16 for VLLM ...');
outputModel = '/outputs/merged'
merged_method = os.environ.get('merged_method')
if merged_method == None or merged_method == '':
    merged_method = 'merged_16bit'
print('save : ' + merged_method + ' ...')
model.save_pretrained_merged(outputModel, tokenizer, save_method=merged_method, )

# 合并模型
# model = model.merge_and_unload()
# model.save_pretrained("/outputs/merged_adapters")

# Merge to 16bit
# model.save_pretrained_merged("/outputs/merged_16bit", tokenizer, save_method = "merged_16bit",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
# model.save_pretrained_merged("/outputs/merged_4bit", tokenizer, save_method = "merged_4bit",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
# model.save_pretrained_merged("/outputs/lora", tokenizer, save_method = "lora",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")

#
# Save to 8bit Q8_0
#
# print('save_pretrained_gguf ...')
# model.save_pretrained_gguf("/outputs/gguf",tokenizer)
