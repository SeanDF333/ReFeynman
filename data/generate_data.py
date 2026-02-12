"""
使用Gemini API生成费曼风格教学数据
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm

# 加载环境变量
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 费曼风格提示词模板
FEYNMAN_PROMPT_TEMPLATE = """
You are Richard Feynman. 
Tone: Conversational, enthusiastic, slightly informal, direct. 
Style: Use "Look," "You see," "It's like this." Use exclamation marks for excitement!

Task: Teach the concept: {topic}

CRITICAL RULES:
1. NO JARGON: If you use a big word, you fail.
2. ANALOGY FIRST: Start with a real-world object (balls, water, rubber bands, colors).
3. "FIRST PRINCIPLES": Don't say "because the formula says so." Say "because the atoms are bumping into each other."
4. ADMIT IGNORANCE: If science doesn't know the answer yet, say "We simply don't know!" (Feynman loved admitting what we don't know).

Format as JSON:
{{
  "topic": "{topic}",
  "student_question": "A natural student question",
  "feynman_response": "Your teaching response (200-400 words)",
  "key_analogy": "The main analogy used",
  "difficulty": "beginner/intermediate/advanced"
}}

Generate ONE complete dialogue now.
"""

# 主题列表
PHYSICS_TOPICS = {
    # ==========================================
    # 1. 经典力学与分析力学 (Mechanics: From Newton to Action)
    # 重点：从“力”的视角转向“能量”和“作用量”的视角
    # ==========================================
    "Classical_Analytical_Mechanics": [
        # 基础直觉
        "Why does a spinning top stay upright? (Gyroscopic Precession)",  # 陀螺进动
        "Coriolis Force: Why do hurricanes spin? (Fictitious Forces)",    # 科里奥利力
        "Coupled Oscillators & Normal Modes",                             # 耦合振子（万物皆弹簧）
        
        # 变分法与最小作用量 (费曼的最爱)
        "The Principle of Least Action",                                  # 最小作用量原理
        "Why nature minimizes Action, not Energy?",                       # 为什么是作用量不是能量？
        "Euler-Lagrange Equation derivation intuition",                   # 欧拉-拉格朗日方程的直觉
        
        # 哈密顿与相空间 (数学物理的桥梁)
        "Legendre Transformation (Geometric meaning)",                    # 勒让德变换的几何意义（切线截距）
        "Why switch from Velocity (Lagrangian) to Momentum (Hamiltonian)?", # 为什么要换变量？
        "Phase Space & Liouville's Theorem",                              # 相空间流体是不可压缩的吗？
        "Poisson Brackets vs Commutators",                                # 泊松括号（通向量子力学的大门）
        "Noether's Theorem: Symmetry implies Conservation",               # 对称性导致守恒律
        "Canonical Transformations"                                       # 正则变换
    ],

    # ==========================================
    # 2. 电磁学与光学 (Electromagnetism & Optics)
    # 重点：场的实在性与波的传播
    # ==========================================
    "Electromagnetism_Optics": [
        # 场的直觉
        "What is a Field? Is it real or mathematical?",                   # 场是真实的吗？
        "Maxwell's Equations: Visualizing Divergence and Curl",           # 散度和旋度的物理图像
        "The Vector Potential (A-field) significance",                    # 矢量势A真的只是数学辅助吗？
        "Lenz's Law: Nature hates change",                                # 楞次定律与惯性
        
        # 介质中的电磁波
        "Why does Light slow down in glass? (Feynman's electron spring model)", # 费曼经典的“电子弹簧模型”解释折射率
        "Kramers-Kronig Relations (Causality)",                           # 因果律与色散
        "Skin Depth in Conductors",                                       # 趋肤效应
        "Retarded Potentials: Why signals can't be instantaneous",        # 推迟势
        
        # 辐射与相对论电动力学
        "Cherenkov Radiation (Sonic boom of light)",                      # 切伦科夫辐射
        "Lienard-Wiechert Potentials",                                    # 运动电荷的势
        "Synchrotron Radiation"                                           # 同步辐射
    ],

    # ==========================================
    # 3. 热力学与统计力学 (Thermodynamics & Stat Mech)
    # 重点：从微观无序导出宏观有序
    # ==========================================
    "Thermodynamics_Statistical": [
        # 熵与不可逆性
        "Maxwell's Demon Paradox",                                        # 麦克斯韦妖
        "The Ratchet and Pawl mechanism",                                 # 棘轮与掣爪（费曼讲义名篇：热涨落）
        "Boltzmann's definition of Entropy (S=klnW)",                     # 熵的微观定义
        "Why Rubber Bands heat up when stretched (Entropic Force)",       # 橡皮筋的熵力
        "The Arrow of Time",                                              # 时间之箭
        
        # 统计分布与系综
        "Canonical Ensemble (Gibbs Distribution)",                        # 正则系综
        "Equipartition Theorem failure at low temps",                     # 能量均分定理的失效（量子化的开端）
        "Black Body Radiation (Ultraviolet Catastrophe)",                 # 紫外灾难
        "Fermi-Dirac vs Bose-Einstein Statistics",                        # 费米子与玻色子的统计差异
        "Bose-Einstein Condensation intuition",                           # 玻色-爱因斯坦凝聚
        "Ising Model (Phase Transitions)"                                 # 伊辛模型与相变
    ],

    # ==========================================
    # 4. 量子力学 (Quantum Mechanics)
    # 重点：波粒二象性、叠加态与路径积分
    # ==========================================
    "Quantum_Mechanics": [
        # 核心概念
        "The Double Slit Experiment with Electrons",                      # 电子双缝干涉（核心谜题）
        "Stern-Gerlach Experiment (Quantized Spin)",                      # 自旋量子化
        "Heisenberg Uncertainty Principle (Fourier Transform connection)", # 不确定性原理与傅里叶变换的关系
        "Schrödinger's Cat (Superposition)",                              # 叠加态
        
        # 形式理论
        "Hilbert Space & Bra-Ket Notation intuition",                     # 希尔伯特空间的几何直觉
        "Operators as Matrices",                                          # 算符即矩阵
        "The Born Rule (Why probability squared?)",                       # 波恩定则
        
        # 费曼的贡献与进阶
        "Feynman Path Integral (Sum over Histories)",                     # 路径积分（历史求和）
        "Aharonov-Bohm Effect (Potentials are real)",                     # AB效应（势是真实的）
        "Quantum Tunneling and Alpha Decay",                              # 量子隧穿
        "Identical Particles (Exchange Symmetry)",                        # 全同粒子与交换对称性
        "Bell's Theorem (Spooky action at a distance)"                    # 贝尔不等式
    ],

    # ==========================================
    # 5. 相对论 (Relativity)
    # 重点：时空结构的改变
    # ==========================================
    "Relativity": [
        # 狭义相对论
        "Michelson-Morley Experiment",                                    # 以太漂移的零结果
        "Simultaneity is relative (Train thought experiment)",             # 同时性的相对性
        "Time Dilation & Muon Decay",                                     # μ子衰变与时间膨胀
        "Twin Paradox resolution",                                        # 双生子佯谬
        "E=mc² derivation intuition",                                     # 质能方程直觉
        "Minkowski Space-Time Diagrams",                                  # 闵可夫斯基时空图
        
        # 广义相对论 (概念层面)
        "Equivalence Principle (Elevator experiment)",                    # 等效原理（电梯实验）
        "Bending of Light by Gravity",                                    # 光线弯曲
        "Gravitational Redshift",                                         # 引力红移
        "Schwarzschild Radius (Black Holes)"                              # 史瓦西半径
    ]
}

# 自动展平为列表
def _flatten_physics_topics():
    flat_list = []
    for category, topics in PHYSICS_TOPICS.items():
        for topic in topics:
            flat_list.append(topic)
    return flat_list

TOPICS = _flatten_physics_topics()



def generate_feynman_dialogue(topic: str, model_name: str = "gemini-2.5-flash") -> dict:
    """使用Gemini生成一条费曼风格对话"""
    model = genai.GenerativeModel(model_name)
    prompt = FEYNMAN_PROMPT_TEMPLATE.format(topic=topic)
    
    try:
        response = model.generate_content(prompt)
        # 提取JSON部分
        text = response.text
        # 去除markdown代码块标记
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        data = json.loads(text.strip())
        return data
    except Exception as e:
        print(f"Error generating for topic '{topic}': {e}")
        return None


def generate_dataset(num_samples: int = 100, output_path: str = "data/feynman_dialogues.json"):
    """批量生成数据集"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    dialogues = []
    topics_to_use = TOPICS * (num_samples // len(TOPICS) + 1)
    topics_to_use = topics_to_use[:num_samples]
    
    print(f"Generating {num_samples} Feynman-style dialogues...")
    
    for topic in tqdm(topics_to_use):
        dialogue = generate_feynman_dialogue(topic)
        if dialogue:
            dialogues.append(dialogue)
            time.sleep(1)  # 避免API限流
    
    # 保存数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dialogues, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated {len(dialogues)} dialogues")
    print(f"📁 Saved to {output_path}")
    
    return dialogues


def convert_to_training_format(dialogues: list, output_path: str = "data/train.jsonl"):
    """转换为HuggingFace训练格式"""
    training_data = []
    
    for d in dialogues:
        # 构建对话格式
        conversation = {
            "messages": [
                {"role": "user", "content": d["student_question"]},
                {"role": "assistant", "content": d["feynman_response"]}
            ],
            "metadata": {
                "topic": d["topic"],
                "analogy": d.get("key_analogy", ""),
                "difficulty": d.get("difficulty", "intermediate")
            }
        }
        training_data.append(conversation)
    
    # 保存为JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Converted to training format: {output_path}")


if __name__ == "__main__":
    # 生成数据
    print("🚀 Starting Feynman dialogue generation with Gemini...")
    
    # 先生成少量测试
    dialogues = generate_dataset(num_samples=500, output_path="data/feynman_dialogues.json")
    
    # 转换为训练格式
    convert_to_training_format(dialogues, output_path="data/train.jsonl")
    
    print("\n✨ Data generation complete!")
    print("Next step: Run SFT training with this data")