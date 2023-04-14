#!/bin/bash
# sh scripts/basics/mnist/train_mnist.sh
# build slurm script
#SBATCH --output=slurm/slurm-%J.out
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH --mem=10G
#SBATCH --time=10:00:00

echo "#!/bin/bash"
module load cuda/11.3
module load anaconda/2022a
export HDF5_USE_FILE_LOCKING='FALSE'



GPU=1
CPU=1
jobname=openood




PYTHONPATH='.':$PYTHONPATH \
srun --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \

python main.py \
--config configs/datasets/mnist/mnist.yml \
configs/preprocessors/base_preprocessor.yml \
configs/networks/lenet.yml \
configs/pipelines/train/baseline.yml
