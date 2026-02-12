"""
Supervised Fine-Tuning (SFT) 训练脚本
使用LoRA在基础模型上进行监督微调
"""

import os
import yaml
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: str = "config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_model_and_tokenizer(config):
    """准备模型和分词器"""
    model_name = config['model']['base_model']
    
    # 4bit量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['model']['load_in_4bit'],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN")
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN")
    )
    
    # 准备模型用于训练
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer


def get_lora_config(config):
    """获取LoRA配置"""
    lora_cfg = config['lora']
    return LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        lora_dropout=lora_cfg['lora_dropout'],
        target_modules=lora_cfg['target_modules'],
        bias=lora_cfg['bias'],
        task_type=lora_cfg['task_type'],
    )


def format_dialogue(example, tokenizer):
    """格式化对话数据"""
    messages = example['messages']
    
    # 构建Qwen格式的对话
    text = ""
    for msg in messages:
        if msg['role'] == 'user':
            text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg['role'] == 'assistant':
            text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
    
    return {"text": text}


def train_sft(config_path: str = "config.yaml"):
    """执行SFT训练"""
    config = load_config(config_path)
    
    print("🚀 Starting SFT training...")
    print(f"📦 Base model: {config['model']['base_model']}")
    
    # 准备模型
    model, tokenizer = prepare_model_and_tokenizer(config)
    
    # 应用LoRA
    lora_config = get_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 加载数据集
    dataset = load_dataset('json', data_files={
        'train': 'data/train.jsonl'
    })
    
    # 格式化数据
    dataset = dataset.map(
        lambda x: format_dialogue(x, tokenizer),
        remove_columns=dataset['train'].column_names
    )
    
    # 训练参数
    sft_cfg = config['sft']
    training_args = TrainingArguments(
        output_dir=config['paths']['output_dir'] + "/sft",
        num_train_epochs=sft_cfg['num_epochs'],
        per_device_train_batch_size=sft_cfg['batch_size'],
        gradient_accumulation_steps=sft_cfg['gradient_accumulation_steps'],
        learning_rate=sft_cfg['learning_rate'],
        warmup_steps=sft_cfg['warmup_steps'],
        logging_steps=sft_cfg['logging_steps'],
        save_steps=sft_cfg['save_steps'],
        max_grad_norm=sft_cfg['max_grad_norm'],
        weight_decay=sft_cfg['weight_decay'],
        fp16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        save_total_limit=3,
        logging_dir=config['paths']['logs_dir'],
        report_to="none",  # 可改为"wandb"
    )
    
    # 创建训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        tokenizer=tokenizer,
        max_seq_length=config['model']['max_seq_length'],
        dataset_text_field="text",
    )
    
    # 开始训练
    print("\n" + "="*50)
    print("🎯 Starting training...")
    print("="*50 + "\n")
    
    trainer.train()
    
    # 保存模型
    final_path = config['paths']['output_dir'] + "/sft_final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"\n✅ SFT training complete!")
    print(f"📁 Model saved to: {final_path}")
    
    return trainer


if __name__ == "__main__":
    train_sft()