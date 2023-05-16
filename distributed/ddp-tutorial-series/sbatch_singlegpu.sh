#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o slurm-sample%j.out
#SBATCH -e slurm-sample%j.err
#SBATCH --gpus 1

# init python environment: copied from ~/.bashrc:
#source /home/yuhui/idiom/bin/activate
eval "$('/opt/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate cnn

echo "which python:"
which python
echo "CUDA_VISIBLE_DEVICES:"
echo $CUDA_VISIBLE_DEVICES

echo "start python script:"
python /home/yuhui/dev/src/github.com/examples/distributed/ddp-tutorial-series/train_singlegpu.py 1 10

echo "done."