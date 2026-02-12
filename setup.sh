#!/bin/bash

# ReFeynman 快速启动脚本

echo "🚀 ReFeynman Setup Script"
echo "=========================="

# 检查conda
if ! command -v conda &> /dev/null
then
    echo "❌ Conda not found. Please install Miniconda first."
    exit 1
fi

# 创建环境
echo "📦 Creating conda environment 'LLM'..."
conda create -n LLM python=3.10 -y

# 激活环境
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate LLM

# 安装依赖
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# 检查.env
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env and add your API keys:"
    echo "   - HF_TOKEN"
    echo "   - GEMINI_API_KEY"
else
    echo "✅ .env file found"
fi

# 创建必要目录
mkdir -p checkpoints logs data/processed

echo ""
echo "✨ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Generate data: python data/generate_data.py"
echo "3. Upload notebooks/colab_training.ipynb to Google Colab"
echo "4. Run training on Colab"
echo ""
echo "Or run locally (if you have GPU):"
echo "   python models/sft_trainer.py"
echo "   python models/grpo_trainer.py"