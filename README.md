# An Empirical Investigation of Early Stopping Optimizations in Proximal Policy Optimization

Source code and reproduction steps for the paper _An Empirical Investigation of Early Stopping Optimizations in Proximal Policy Optimization_.

## Training

### 1. Dependencies

The list of python packages required to run the experiments can be found in the `requirements.txt`.
We recommend creating a dedicated virtual environment with the Anaconda Python package manager.

### 2. Weights And Biases (WANDB) Setup

To log the experiments results and use the scripts provided in the `evaluation` folder to generate the corresponding plots, we
recommend using the WANDB service.
The `ppo2_continuous_action.py` already contains the necessary logic.
All that is left to do is to set your WANDB API KEY as environment variable, namely with:
```
export WANDB_API_KEY=<your api key here>
```
The API key is found [here](https://app.wandb.ai/settings).

### 3. Running the experiments

We have provided the scripts that contained the 1210 experiments ran for this study as `all.sh`.
Depending on your server infrastructure, there might be a need to run only a few experiments at the time (for example, by first commenting the whole file, then progressively uncommenting and running only a few blocks at the same time.)

For a more streamlined training process, we suggest using services such AWS Batch, which automates the job execution.

## Evaluation

The results of the experiments can be accessed in an interactive way using the following [Weights and Biases Project](ttps://wandb.ai/cleanrl/ppo-kle).

Furthermore, the various plots and tables used in the paper can be generated using the `AllPlots` Jupyter Notebook located in the `evaluation` folder.
To quickly generate the various plots, we also provided pre-cached data corresponding to the `ppo-early-stopping` as the `evaluation/clearl_ppo_kle` folder.
To regenerate the data used for the plots, please remove the folder specified above and rerun the notebook. However, depending on the machine used, the data can take up to a few hours to generate.

In case you have you have followed the "Training" section and generated your experiments under a different WANDB project, please edit the `wandb_entity_project` variable in the first cell of the notebook and re-run it to obtain the corresponding plots as in the paper.
