#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o slurm-sample%j.out
#SBATCH -e slurm-sample%j.err
#SBATCH --gpus 2

# init python environment: copied from ~/.bashrc:
#source /home/yuhui/idiom/bin/activate
# eval "$('/opt/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# conda activate cnn
docker run -d --name dlprof_multi --cap-add=SYS_ADMIN --gpus=2 --shm-size=1g --ulimit memlock=-1  --ulimit stack=67108864 -it -v $(pwd):/host_pwd nvcr.io/nvidia/pytorch_dlprof:23.03-py3

echo "which python:"
which python
echo "CUDA_VISIBLE_DEVICES:"
echo $CUDA_VISIBLE_DEVICES

echo "start python script:"
docker exec -w /host_pwd dlprof_multi dlprof --mode=pytorch python train_multigpu_dl.py 1 10

echo "done."