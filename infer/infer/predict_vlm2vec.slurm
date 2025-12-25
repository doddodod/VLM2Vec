#!/bin/bash
#SBATCH --job-name=embedding_infer
#SBATCH -A hexm-critical
#SBATCH --partition=critical,hexm
#SBATCH --output=/public/home/wangby2025/plusLab/slurmLog/37k_simi_bestE_embedding_infer_%j.out
#SBATCH --error=/public/home/wangby2025/plusLab/slurmLog/37k_simi_bestE_embedding_infer_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --nodelist=ai_gpu34 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wangby2025@shanghaitech.edu.cn



echo "🔧 激活conda环境..."
module purge
module load anaconda/3
source /public/home/wangby2025/anaconda3/bin/activate vlm2vec 

# 切换到项目目录
cd /public/home/wangby2025/plusLab/VLM2Vec || { echo "❌ 项目目录不存在"; exit 1; }


export FLASH_ATTENTION_DISABLE=0
export USE_FLASH_ATTENTION=1
export FLASH_ATTENTION_FORCE_BUILD=0
export FLASH_ATTENTION_SKIP_CUDA_BUILD=0

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"


python /public/home/wangby2025/plusLab/VLM2Vec/infer/infer/predict_scene_graph_recall.py


