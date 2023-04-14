echo $DIR_PATH
# sh ./scripts/download/dowanload.sh
python /home/gridsan/nhulkund/OpenOOD/scripts/download/dowanload.py \
--contents 'datasets' 'checkpoints' \
--datasets 'default' \
--checkpoints 'all' \
--save_dir '/home/gridsan/nhulkund/OpenOOD/scripts/download/data' '/home/gridsan/nhulkund/OpenOOD/scripts/download/results' \
--dataset_mode 'benchmark'
