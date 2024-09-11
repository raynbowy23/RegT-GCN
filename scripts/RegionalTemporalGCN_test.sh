# python3 predict.py --num_timesteps_in 6 --num_timesteps_out 1 --tr 0.2 \
#     --model 'RegionalTemporalGCN' --tf occrate --visualize false --pretrained_idx 50
python3 predict.py --num_timesteps_in 6 --num_timesteps_out 3 --tr 0.2 \
    --model 'RegionalTemporalGCN' --tf occrate --visualize false --pretrained_idx 50 # --pretrained_model 'model_in6_out3_epoch30.pt'
# python3 predict.py --num_timesteps_in 6 --num_timesteps_out 12 --tr 0.2 \
#     --model 'RegionalTemporalGCN' --tf occrate --visualize false --pretrained_idx 50
# python3 predict.py --num_timesteps_in 6 --num_timesteps_out 36 --tr 0.2 \
#     --model 'RegionalTemporalGCN' --tf occrate --visualize false --pretrained_idx 50 # --pretrained_model 'model_in6_out3_epoch30.pt'
    