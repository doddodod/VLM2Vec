#!/bin/bash
#SBATCH --job-name=stage3
#SBATCH -A hexm
#SBATCH --partition=hexm_l40
#SBATCH --output=/public/home/xiaojw2025/Workspace/RAHP/slurm/logs/stage3_%j.out
#SBATCH --error=/public/home/xiaojw2025/Workspace/RAHP/slurm/logs/stage3_%j.err
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=24
#SBATCH --mem=450G
#SBATCH --gres=gpu:4
#SBATCH --mail-type=all
#SBATCH --mail-user=mxzxx093834@163.com

# 第三阶段：使用Stage2生成的加权描述进行embedding相似度匹配
# 实验组4：加权Stage2描述（Top-K，包含相似度分数和谓词名称）


set -e
source ~/.bashrc 

# 记录启动时间
START_TIME=$(date +%s)
START_TIME_STR=$(date "+%Y-%m-%d %H:%M:%S")

echo "================================================================================"
echo "🚀 Stage3 Embedding相似度匹配任务启动"
echo "================================================================================"
echo "📅 启动时间: $START_TIME_STR"
echo "📝 Job ID: $SLURM_JOB_ID"
echo "💻 Node: $SLURM_NODELIST"
echo "🎯 GPU数量: $SLURM_GPUS_ON_NODE"
echo "⚙️  任务数: $SLURM_NTASKS"
echo "📂 工作目录: $(pwd)"
echo "👤 用户: $(whoami)"
echo "================================================================================"

# 激活conda环境
echo "🔧 激活conda环境..."
source /public/home/xiaojw2025/anaconda3/etc/profile.d/conda.sh
conda activate qwen || { echo "❌ 无法激活环境"; exit 1; }

# 检查GPU状态
echo "🔍 检查GPU状态..."
nvidia-smi
echo "======================================="
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 配置Flash Attention环境变量（如果GPU支持）
echo "🔧 配置Flash Attention..."
export FLASH_ATTENTION_DISABLE=0
export USE_FLASH_ATTENTION=1
export FLASH_ATTENTION_FORCE_BUILD=0
export FLASH_ATTENTION_SKIP_CUDA_BUILD=0
echo "   Flash Attention环境变量已设置"

# 切换到工作目录
cd /public/home/xiaojw2025/Workspace/VLM2Vec || { echo "❌ 无法切换到工作目录"; exit 1; }

# 执行推理（使用4个GPU）
PYTHON_SCRIPT="/public/home/xiaojw2025/Workspace/VLM2Vec/embedding/infer/stage3.py"

echo ""
echo "================================================================================"
echo "🚀 开始执行Stage3推理"
echo "================================================================================"
echo "📜 执行脚本: $PYTHON_SCRIPT"
echo "🎯 GPU数量: 4"
echo "🐍 Python版本: $(python --version 2>&1)"
echo "🔧 Conda环境: $CONDA_DEFAULT_ENV"
echo "📂 工作目录: $(pwd)"
echo "⏰ 执行开始时间: $(date "+%Y-%m-%d %H:%M:%S")"
echo "================================================================================"
echo ""

# 执行Python脚本（使用4个GPU）
python $PYTHON_SCRIPT --num_gpus 4 

# 记录结束时间和耗时
END_TIME=$(date +%s)
END_TIME_STR=$(date "+%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED_TIME / 3600))
ELAPSED_MINS=$(((ELAPSED_TIME % 3600) / 60))
ELAPSED_SECS=$((ELAPSED_TIME % 60))

echo ""
echo "================================================================================"
echo "✅ Stage3推理任务完成"
echo "================================================================================"
echo "⏰ 开始时间: $START_TIME_STR"
echo "⏰ 结束时间: $END_TIME_STR"
echo "⏱️  总耗时: ${ELAPSED_HOURS}小时 ${ELAPSED_MINS}分钟 ${ELAPSED_SECS}秒 "
echo "📝 Job ID: $SLURM_JOB_ID"
echo "================================================================================"