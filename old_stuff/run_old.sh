# assuming that the data is already created

export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export XLA_USE_BF16=1

rm -r final_model_old
mv final_model final_model_old
mkdir final_model
mv runs "runs-$(date)" 

# nohup python3 train.py --data_folder="val-4/" --checkpoint_folder="checkpoints/" \
#         --output_folder="final_model/" --train_batch_size=12 --valid_batch_size=128 \
#         --epoch=5 --output_labels=4 --num_cores=8 --num_workers=4 --log_steps=1000 \
#         --comment="first run, recsys2020 dev only, multi-head" --transformer_nheads=4 --start_epoch=3 --load_checkpoint=True&

nohup python3 train_old.py --data_folder="val-6/" --checkpoint_folder="checkpoints/" \
    --output_folder="final_model/" --train_batch_size=8 --valid_batch_size=128 \
    --epoch=5 --output_labels=4 --num_cores=8 --num_workers=4 --log_steps=1000 \
    --comment="first run, recsys2020 dev only, multi-head" --transformer_nheads=4 &
