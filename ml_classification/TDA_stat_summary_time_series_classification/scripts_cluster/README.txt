

## ON HPC 

On HPC, first run `scripts_cluster/submit_config_search.sh`. After completing these jobs which run as array jobs
Then, generate the best config files, using `sh submit_rebuild_best_config.sh` in `hparams...` directory
Having the best_config.jason and best_config_small.json files, run `scripts_cluster/submit_best_training.sh`

Before running `scripts_cluster/submit_best_training.sh`make sure to increase patience and num_epoch 
and time_duration (do not touch segmentation duration) in the best_config*.json files.

To obtain noise-robustness related results, run `scripts_cluster/submit_raw_signal_robustness.sh' outside `seg...' directories.


