import os

hf_token = os.environ.get('HF_TOKEN')
hf_token = None if hf_token == None else hf_token

org_model_name = os.environ.get('model_name')
org_model_name = 'unsloth/llama-3-8b-bnb-4bit' if org_model_name == None else org_model_name

#
# 加载模型
#
from unsloth import FastLanguageModel
import torch

print('load model : ' + org_model_name)
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

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
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token=hf_token
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
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
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

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=4000, #60
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

#
# 验证推理
#
print('train ...')
trainer_stats = trainer.train(resume_from_checkpoint = True) #检查点开始训练
print(trainer_stats)

#
# 推理模型
#
print('train ...')
# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Continue the fibonnaci sequence.",  # instruction
            "1, 1, 2, 3, 5, 8",  # input
            "",  # output - leave this blank for generation!
        )
    ], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
tokenizer.batch_decode(outputs)

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
