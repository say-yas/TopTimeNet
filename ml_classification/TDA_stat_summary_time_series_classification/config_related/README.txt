
To merely generate a config file, use  
`python config_file_generation.py`

To optimize configuration parameters, follow these steps

Step 1 — generate the search space config (once, then edit) [Check path to save the config file]
`python create_search_config.py`

Step 2 — run the search (default: 30 random trials)
`python hparam_search.py`

After obtainin the best found config, try

python main_train_time_series_transformer.py --config test/hparam_search_results/best_config.json

## ON HPC
On HPC, first run `scripts_cluster/submit_config_search.sh`. After completing these jobs which run as array jobs and having the best_config.jason file, , run `scripts_cluster/submit_best_training.sh`
If the first script couldn't generatethe best config files, `sh submit_rebuild_best_config.sh`