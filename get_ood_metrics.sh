#!/bin/bash

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

python get_ood_metrics.py \
 --ood_metric "${1}" \
 --dataset "${2}" \
 --rand_augment "${3}" \
 --rand_n ${4} \
 --rand_m ${5} \





