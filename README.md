# ReFeynman 🎓⚛️

> **"If you can't explain it simply, you don't understand it well enough."** - Richard Feynman

A **physics-focused** educational LLM fine-tuned with **GRPO (Group Relative Policy Optimization)** to teach complex physics concepts in the Feynman style: simple, intuitive, and using everyday analogies.

**Special Focus**: Core concepts from *The Feynman Lectures on Physics*, including the Ratchet and Pawl, Path Integrals, and the Electron Spring Model.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

## 🌟 Features

- **GRPO Training**: Implementation of Group Relative Policy Optimization for alignment
- **Feynman-Style Physics Teaching**: Trained on ~75 carefully selected physics concepts from classical mechanics to quantum field theory
- **Feynman Lectures Integration**: Includes classic examples like the Ratchet and Pawl, Electron Spring Model, and Path Integral formulation
- **Efficient Fine-tuning**: Uses LoRA (Low-Rank Adaptation) for parameter-efficient training
- **Free Training Pipeline**: Runs entirely on Google Colab free tier (T4 GPU)
- **Gemini-Powered Data**: Uses Gemini API for reward modeling and physics dialogue generation

## 🏗️ Architecture

```
Base Model: Qwen2.5-7B-Instruct
    ↓
SFT (Supervised Fine-Tuning) with LoRA
    ↓
GRPO (Group Relative Policy Optimization)
    ↓ 
ReFeynman Model
```

### Training Pipeline

1. **Data Generation**: Gemini 1.5 Flash generates Feynman-style teaching dialogues
2. **SFT Phase**: LoRA fine-tuning on educational conversations
3. **GRPO Phase**: Reinforcement learning with Gemini as reward model
4. **Evaluation**: Multi-metric assessment of teaching quality

## 📊 GRPO Implementation

Our GRPO implementation follows the [original paper](https://arxiv.org/abs/2402.03300):

```python
# For each prompt:
1. Generate K responses from policy model
2. Score each response with reward model
3. Compute group-relative advantages:
   advantage_i = (reward_i - mean(rewards)) / std(rewards)
4. Update policy with PPO-style objective:
   L = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] - β * KL
```

**Key Components**:
- **Group Sampling**: 4 responses per prompt for stable advantage estimation
- **Gemini Reward Model**: Evaluates teaching quality (clarity, analogies, accuracy)
- **KL Penalty**: Prevents model from deviating too far from SFT checkpoint

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google Colab account (for free GPU training)
- HuggingFace account
- Gemini API key (free tier)

### Installation

```bash
# Clone the repository
git clone https://github.com/SeanDF333/ReFeynman.git
cd ReFeynman

# Create conda environment
conda create -n LLM python=3.10
conda activate LLM

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# - HF_TOKEN: HuggingFace token
# - GEMINI_API_KEY: Gemini API key
```

### Training (on Colab)

1. Upload `notebooks/colab_training.ipynb` to Google Colab
2. Set runtime to GPU (T4)
3. Add secrets in Colab:
   - `HF_TOKEN`
   - `GEMINI_API_KEY`
4. Run all cells

**Expected Training Time**:
- Data Generation: ~30 minutes (500 dialogues)
- SFT Training: ~2-3 hours
- GRPO Training: ~1-2 hours

### Local Testing

```bash
# Interactive mode
python demo.py --checkpoint checkpoints/grpo_final

# Single question
python demo.py --question "Explain quantum entanglement simply"
```

## 📁 Project Structure

```
ReFeynman/
├── data/
│   ├── generate_data.py      # Gemini-based data generation
│   └── train.jsonl            # Training data (generated)
├── models/
│   ├── sft_trainer.py         # Supervised fine-tuning
│   └── grpo_trainer.py        # GRPO implementation ⭐
├── evaluation/
│   └── metrics.py             # Evaluation metrics
├── notebooks/
│   └── colab_training.ipynb   # End-to-end Colab notebook
├── checkpoints/               # Model checkpoints (gitignored)
├── config.yaml                # Training configuration
├── demo.py                    # CLI demo
└── README.md
```

## 🎯 Results

### Sample Outputs

**Question**: "Can you explain quantum entanglement like I'm 10 years old?"

**Before GRPO (SFT only)**:
> Quantum entanglement is a phenomenon where particles become correlated...

**After GRPO**:
> Imagine you have two magic coins. When you flip one and it lands on heads, 
> the other coin INSTANTLY lands on tails - even if they're on opposite sides 
> of the universe! That's quantum entanglement. The particles are "connected" 
> in a spooky way, as Einstein called it...

### Training Metrics

| Metric | SFT | GRPO |
|--------|-----|------|
| Avg Reward | 0.62 | 0.81 |
| Clarity Score | 6.8/10 | 8.4/10 |
| Analogy Usage | 45% | 78% |

*(Metrics evaluated on 100 test questions)*

## 🛠️ Technical Details

### Hyperparameters

**SFT**:
- LoRA rank: 16
- Learning rate: 2e-4
- Batch size: 4 (grad accum: 4)
- Epochs: 3

**GRPO**:
- Samples per prompt: 4
- Learning rate: 5e-6
- KL coefficient: 0.05
- Clip range: 0.2
- Iterations: 5

### Compute Requirements

- **Training**: Free Google Colab T4 (16GB VRAM)
- **Inference**: 1650Ti (4GB) with 4-bit quantization
- **Total Training Cost**: $0 (using free tiers)

## 🔮 Future Work

- [ ] Implement ORPO/DAPO for comparison
- [ ] Multi-modal teaching (diagrams, animations)
- [ ] Adaptive difficulty based on user level
- [ ] Expand to 30+ subjects
- [ ] Fine-grained reward models (per subject)
- [ ] Gradio web interface
- [ ] Retrieval-augmented teaching (RAG)

## 📚 References

1. [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300) - GRPO Paper
2. [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
3. [Qwen2.5 Technical Report](https://arxiv.org/abs/2309.16609)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Richard Feynman for inspiring a generation of teachers
- Anthropic for Claude and research guidance
- Google for Gemini API and Colab
- HuggingFace for transformers and model hosting
- Alibaba for Qwen models

## 📧 Contact

For questions or collaborations, please open an issue or contact [xiaoyanfan333@gmail.com]

---

⭐ If you find this project helpful, please consider starring it!
