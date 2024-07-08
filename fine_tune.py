from unsloth import FastLanguageModel
import gc
import torch

torch.cuda.empty_cache()
gc.collect()

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",  # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",  # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",  # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",  # Gemma 2.2x faster!
]  # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_, # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

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
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset

torch.cuda.empty_cache()
gc.collect()

dataset = load_dataset("./data", split="train")
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=48,
    packing=False,  # 启用 packing 来提高效率
    args = TrainingArguments(
        per_device_train_batch_size = 2,  # 增加到 GPU 能處理的最大值
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.03,  # 使用比例而不是固定步數
        num_train_epochs=1,
        max_steps=20000,
        learning_rate = 1e-3,
        fp16 = True,
        logging_steps = 5000,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        gradient_checkpointing = True,  # 添加梯度檢查點
    ),
)

torch.cuda.empty_cache()
gc.collect()

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

trainer.model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

#! throw everything on Colab, because local machine has no enough power

# #@title Show final memory and time stats
# used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
# used_percentage = round(used_memory         /max_memory*100, 3)
# lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
# print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
# print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
# print(f"Peak reserved memory = {used_memory} GB.")
# print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
# print(f"Peak reserved memory % of max memory = {used_percentage} %.")
# print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# torch.cuda.empty_cache()
# gc.collect()

# # 重新加載模型
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="./saved_model",
#     max_seq_length=max_seq_length,
#     dtype=dtype,
#     load_in_4bit=load_in_4bit
# )
# FastLanguageModel.for_inference(model) # Enable native 2x faster inference
# inputs = tokenizer(
# [
#     alpaca_prompt.format(
#         "Continue the fibonnaci sequence.", # instruction
#         "1, 1, 2, 3, 5, 8", # input
#         "", # output - leave this blank for generation!
#     )
# ], return_tensors = "pt").to("cuda")

# outputs = model.generate(
#     **inputs,
#     max_new_tokens=64,
#     use_cache=True
# )
# tokenizer.batch_decode(outputs)

# from transformers import TextStreamer
# text_streamer = TextStreamer(tokenizer)
# _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

# model.save_pretrained("lora_model") # Local saving
# tokenizer.save_pretrained("lora_model")
# # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

# if False:
#     from unsloth import FastLanguageModel
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
#         max_seq_length = max_seq_length,
#         dtype = dtype,
#         load_in_4bit = load_in_4bit,
#     )
#     FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# # alpaca_prompt = You MUST copy from above!

# inputs = tokenizer(
# [
#     alpaca_prompt.format(
#         "What is a famous tall tower in Paris?", # instruction
#         "", # input
#         "", # output - leave this blank for generation!
#     )
# ], return_tensors = "pt").to("cuda")

# outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
# tokenizer.batch_decode(outputs)

# if False:
#     # I highly do NOT suggest - use Unsloth if possible
#     from peft import AutoPeftModelForCausalLM
#     from transformers import AutoTokenizer
#     model = AutoPeftModelForCausalLM.from_pretrained(
#         "lora_model", # YOUR MODEL YOU USED FOR TRAINING
#         load_in_4bit = load_in_4bit,
#     )
#     tokenizer = AutoTokenizer.from_pretrained("lora_model")

#     # Merge to 16bit
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# # Merge to 4bit
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# # Just LoRA adapters
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")

# # Save to 8bit Q8_0
# if False: model.save_pretrained_gguf("model", tokenizer,)
# if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# # Save to 16bit GGUF
# if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
# if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# # Save to q4_k_m GGUF
# if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
# if True: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# import shutil

# # 本地文件路徑
# local = '/content/model-unsloth.Q4_K_M.gguf'

# # 目標 Google Drive 路徑
# google_drive = '/content/drive/My Drive/model-unsloth.Q4_K_M.gguf'

# # 複製文件到 Google Drive
# shutil.copy(local, google_drive)

# print('複製成功！')