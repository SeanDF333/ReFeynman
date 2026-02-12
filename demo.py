"""
ReFeynman - 命令行演示
快速测试训练好的模型
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def load_model(checkpoint_path: str):
    """加载模型"""
    print(f"📦 Loading model from {checkpoint_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    
    print("✅ Model loaded!\n")
    return model, tokenizer


def generate_response(model, tokenizer, question: str, max_length: int = 400):
    """生成回答"""
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return response


def interactive_mode(model, tokenizer):
    """交互模式"""
    print("="*60)
    print("🎓 ReFeynman - Interactive Mode")
    print("="*60)
    print("Ask me anything about physics, math, or science!")
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        print("\nFeynman: ", end="", flush=True)
        response = generate_response(model, tokenizer, question)
        print(response)
        print("\n" + "-"*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ReFeynman CLI Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/grpo_final",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to ask (non-interactive)"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model(args.checkpoint)
    
    if args.question:
        # 单次问答
        response = generate_response(model, tokenizer, args.question)
        print(f"\nQ: {args.question}")
        print(f"\nA: {response}\n")
    else:
        # 交互模式
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()