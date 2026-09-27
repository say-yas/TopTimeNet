To merely generate a config file, use
`python config_file_generation.py`

To optimize configuration parameters, follow these steps:

Step 1: generate the search space config (once, then edit). Check the
path to save the config file.
`python create_search_config.py`

Step 2: run the search (default: 30 random trials)
`python search_best_config.py`

After obtaining the best found config, try:

`python main_train_time_series_tda_stat_summary.py --config test/hparam_search_results/best_config.json`

## ON HPC

On HPC, first run `scripts_cluster/submit_config_search.sh`. After these
array jobs complete and you have the best_config.json file, run
`scripts_cluster/submit_best_training.sh`.

If the first script couldn't generate the best config file, run
`sh submit_rebuild_best_config.sh`.