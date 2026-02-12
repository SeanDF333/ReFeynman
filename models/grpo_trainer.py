"""
GRPO (Group Relative Policy Optimization) 实现
论文复现: https://arxiv.org/abs/2402.03300

核心思想:
1. 为每个prompt生成多个response
2. 使用reward model对responses打分
3. 计算group内的相对优势(advantage)
4. 用PPO-style更新策略
"""

import os
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


@dataclass
class GRPOConfig:
    """GRPO配置"""
    num_iterations: int = 5
    num_samples_per_prompt: int = 4
    batch_size: int = 2
    learning_rate: float = 5e-6
    kl_coef: float = 0.05
    clip_range: float = 0.2
    gamma: float = 1.0
    reward_scale: float = 1.0
    max_new_tokens: int = 512


class GeminiRewardModel:
    """使用Gemini作为reward model评估教学质量"""
    
    REWARD_PROMPT = """You are evaluating a teaching response for quality and clarity.

Student Question: {question}
Teaching Response: {response}

Rate this response on a scale of 0-10 based on:
1. Clarity and simplicity (Feynman-style)
2. Use of analogies and examples
3. Accuracy of content
4. Engagement and enthusiasm
5. Answering the question directly

Respond with ONLY a number between 0 and 10. No explanation.
"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model = genai.GenerativeModel(model_name)
    
    def get_reward(self, question: str, response: str) -> float:
        """获取单个response的奖励分数"""
        try:
            prompt = self.REWARD_PROMPT.format(
                question=question,
                response=response
            )
            result = self.model.generate_content(prompt)
            score = float(result.text.strip())
            return score / 10.0  # 归一化到[0, 1]
        except Exception as e:
            print(f"Reward error: {e}")
            return 0.5  # 默认中等分数
    
    def get_batch_rewards(self, questions: List[str], responses: List[str]) -> List[float]:
        """批量获取奖励"""
        rewards = []
        for q, r in zip(questions, responses):
            rewards.append(self.get_reward(q, r))
        return rewards


class GRPOTrainer:
    """GRPO训练器"""
    
    def __init__(
        self,
        model,
        tokenizer,
        reward_model: GeminiRewardModel,
        config: GRPOConfig,
        ref_model=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.config = config
        self.ref_model = ref_model if ref_model else model
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )
    
    def generate_responses(self, prompts: List[str]) -> List[List[str]]:
        """为每个prompt生成多个responses"""
        all_responses = []
        
        for prompt in prompts:
            responses = []
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            for _ in range(self.config.num_samples_per_prompt):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                responses.append(response)
            
            all_responses.append(responses)
        
        return all_responses
    
    def compute_advantages(self, rewards: List[List[float]]) -> List[List[float]]:
        """计算group relative advantages"""
        advantages = []
        
        for group_rewards in rewards:
            # 组内归一化
            mean_reward = sum(group_rewards) / len(group_rewards)
            std_reward = (sum((r - mean_reward) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
            std_reward = max(std_reward, 1e-8)  # 避免除零
            
            group_advantages = [(r - mean_reward) / std_reward for r in group_rewards]
            advantages.append(group_advantages)
        
        return advantages
    
    def compute_kl_divergence(self, log_probs, ref_log_probs):
        """计算KL散度"""
        return (log_probs.exp() * (log_probs - ref_log_probs)).sum(dim=-1).mean()
    
    def train_step(self, prompts: List[str], questions: List[str]):
        """单步GRPO训练"""
        # 1. 生成responses
        all_responses = self.generate_responses(prompts)
        
        # 2. 获取rewards
        all_rewards = []
        for i, responses in enumerate(all_responses):
            rewards = self.reward_model.get_batch_rewards(
                [questions[i]] * len(responses),
                responses
            )
            all_rewards.append(rewards)
        
        # 3. 计算advantages
        advantages = self.compute_advantages(all_rewards)
        
        # 4. 策略更新
        total_loss = 0
        for prompt, responses, advs in zip(prompts, all_responses, advantages):
            for response, advantage in zip(responses, advs):
                # 构建完整序列
                full_text = prompt + response
                inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
                
                # 前向传播
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                log_probs = F.log_softmax(outputs.logits, dim=-1)
                
                # 参考模型log probs (for KL penalty)
                with torch.no_grad():
                    ref_outputs = self.ref_model(**inputs, labels=inputs['input_ids'])
                    ref_log_probs = F.log_softmax(ref_outputs.logits, dim=-1)
                
                # PPO-style loss
                ratio = (log_probs - ref_log_probs).exp().mean()
                clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
                
                policy_loss = -torch.min(
                    ratio * advantage,
                    clipped_ratio * advantage
                )
                
                # KL penalty
                kl_loss = self.compute_kl_divergence(log_probs, ref_log_probs)
                
                # 总损失
                loss = policy_loss + self.config.kl_coef * kl_loss
                total_loss += loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'avg_reward': sum(sum(r) for r in all_rewards) / sum(len(r) for r in all_rewards),
            'avg_advantage': sum(sum(a) for a in advantages) / sum(len(a) for a in advantages)
        }
    
    def train(self, train_dataset, num_iterations: int = None):
        """执行完整GRPO训练"""
        if num_iterations is None:
            num_iterations = self.config.num_iterations
        
        print(f"\n🚀 Starting GRPO training for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            print(f"\n{'='*50}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*50}")
            
            # 随机采样batch
            batch_indices = torch.randperm(len(train_dataset))[:self.config.batch_size]
            batch = [train_dataset[int(i)] for i in batch_indices]
            
            prompts = [item['messages'][0]['content'] for item in batch]
            questions = prompts  # 学生问题
            
            # 训练步骤
            metrics = self.train_step(prompts, questions)
            
            print(f"📊 Loss: {metrics['loss']:.4f}")
            print(f"🎯 Avg Reward: {metrics['avg_reward']:.4f}")
            print(f"📈 Avg Advantage: {metrics['avg_advantage']:.4f}")
        
        print("\n✅ GRPO training complete!")


def load_model_for_grpo(sft_checkpoint_path: str, config: dict):
    """加载SFT后的模型用于GRPO训练"""
    base_model = config['model']['base_model']
    
    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN")
    )
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(model, sft_checkpoint_path)
    
    tokenizer = AutoTokenizer.from_pretrained(
        sft_checkpoint_path,
        trust_remote_code=True
    )
    
    return model, tokenizer


def main():
    """主训练流程"""
    # 加载配置
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载SFT模型
    sft_path = config['paths']['output_dir'] + "/sft_final"
    print(f"📦 Loading SFT model from {sft_path}")
    model, tokenizer = load_model_for_grpo(sft_path, config)
    
    # 创建reward model
    reward_model = GeminiRewardModel()
    
    # 加载训练数据
    dataset = load_dataset('json', data_files={'train': 'data/train.jsonl'})
    
    # GRPO配置
    grpo_cfg = config['grpo']
    grpo_config = GRPOConfig(
        num_iterations=grpo_cfg['num_iterations'],
        num_samples_per_prompt=grpo_cfg['num_samples_per_prompt'],
        batch_size=grpo_cfg['batch_size'],
        learning_rate=grpo_cfg['learning_rate'],
        kl_coef=grpo_cfg['kl_coef'],
        clip_range=grpo_cfg['clip_range'],
    )
    
    # 创建训练器
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        config=grpo_config
    )
    
    # 训练
    trainer.train(dataset['train'])
    
    # 保存模型
    output_path = config['paths']['output_dir'] + "/grpo_final"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"💾 Model saved to {output_path}")


if __name__ == "__main__":
    main()