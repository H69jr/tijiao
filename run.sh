#!/bin/bash
#SBATCH --job-name=winograd_vgg_hjr
#SBATCH --output=winograd_%j.out
#SBATCH --error=winograd_%j.err
#SBATCH --partition=Compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --mem=8G

module purge
module load gcc/11.4.0

# 设置OpenMP线程数
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# 打印环境信息
echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "Running on $(hostname)"

# 运行Winograd测试
./winograd conf/vgg.conf

# 性能分析
# nsys profile --stats=true ./winograd conf/vgg.conf
