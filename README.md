# Content Aware Recommending Systems with Graph Convolutional Embeddings (CARS-GCE)
#### Double Degree in Mathematics and Computer Science, Universitat de Barcelona
#### Degree's Final Project (TFG), 2020-2021

Project available online: https://github.com/lorenzovigo/TFG/
_____

This project consists in a **Recommender System** that will allow us to compare the performance of different models (**Matrix Factorization** and **Factorization Machines**) set in their optimal hyperparamaters (using **HyperOpt** in order to find them), with the added possibility of applying extensions to said models, such as the use of **Graph Convolutional Embeddings** and online information to make our system more **context aware**.

You'll find two main folders in this repository's root: `code` (which contains the whole project) and `data` (where datasets should be included). We'll now explain how to execute all of the project's workflows.

## **Get your datasets ready**

### **Download your datasets (required)**

_____

First of all, clone the repository locally.

Then, **download at least one of the datasets** available for model training. We can do this by **executing the following command in `code` folder**:

	python dataset_downloader.py

This script will download the **MovieLens 100k dataset**. In case you want to download the **MovieLens 1M dataset**, you may use the `dataset` flag this way:

	python dataset_downloader.py --dataset=ml-1m

### **Extend your datasets (optional)**

_____

In case you are willing to **use online data to extend the previous datasets**, you should **execute the following command in `code` folder**:

	python dataset_extender.py --api_key=<api_key>

`<api_key>` should be substituted by a **MovieDB API Key** owned by you (more information here: https://developers.themoviedb.org/3/getting-started/introduction). 

This command will download information about movie **genres and actors** for all the movies in ml-100k dataset. **We highly recommend executing this script with hours of anticipation**, since depending on factors such as your internet connection and the dataset you are extending, it may take **several hours to complete and it should not be interrupted.**

Should you interrupt the process, don't you worry: the script will be much faster while processing movies which data has already been downloaded from the API.

**Available arguments:**

`--dataset=ml-1m`

Will switch the dataset you are extending. Example:

	python dataset_extender.py --dataset=ml-1m

`--no_genres`

Skips downloading movie genre information while extending the dataset. Examples:

	python dataset_extender.py --no_genres
	python dataset_extender.py --dataset=ml-1m --no_genres

`--no_actors`

Works the same way as the previous command, but it's used in order to skip downloading information about the actors taking part in the movies.


### **Post-process your datasets (optional)**

_____

We include a dataset to **filter out actors with few appearances** along the movies included in the datasets. **This script shouldn't be executed unless you extended your datasets first**. Execute the following command in the `code` folder to post-process the MovieLens-100k dataset: 

	python dataset_postprocessor.py

`--dataset=ml-1m`

Will switch the dataset you are post-processing. Example:

	python dataset_postprocessor.py --dataset=ml-1m

`--min_actor_appearances=<value>`

Where `<value>` is an integer, number of movies the actor should take part in in the dataset in order to be considered relevant enough to be included in the extension. Default value is 10.


## **Tune your model settings (optional)**

The included script `tune.py` implements the search of **optimal parameters** for each model using **Bayesian Optimization**. You may run this script by executing the following command in the `code` folder:

	python tune.py

This script has several available arguments, which are explained at the end of this file. The results will be saved in the `tune_logs` folder.

## **Main execution (required)**

The **main script** is in charge of **splitting** the available datasets (which may be extended or not and post-processed or not), performing **negative sampling**, building the adjacency matrix, **building the graph structure** (if needed), initializing and **training the models** and **evaluating their performances**. This pipeline can be executed by running the following command in the `code` folder:

	python main.py

This script has several available arguments, which are detailed below.
_____

**Relevant arguments for `main.py` and `tune.py` (optional)**

You can always run this command in order to get the most updated help on arguments:

```
$ python main.py --help
```

Each argument will be followed by some examples, which do not represent all their use cases.

`--dataset=ml-1m`

Allows to use models with the **MovieLens 1M** dataset.

	python main.py --dataset=ml-1m
	python tune.py --dataset=ml-1m

`--algo_name=<value>`

Defines the model that is either being used or that is being tuned. Default `<value>` is `mf`, while `fm` is also available. They respectively refer to **Matrix Factorization** and **Factorization Machines**.

	python main.py --algo_name=fm
	python main.py --dataset=ml-1m --algo_name=mf
	python tune.py --algo_name=fm

`--context`

Disables the addition of context to the model.

	python main.py --dataset=ml-1m --context
	python tune.py --context
	
`--gce`

**Enables the GCE layer** in the selected model, which substitutes the Embedding Layer included in them.

	python main.py --gce
	python main.py --algo_name=fm --gce
	python tune.py --algo_name=fm --gce
	
`--genres` and `--actors`

Respectively, adds **genres and actors as side-information** to the model. **These arguments require having extended the selected dataset accordingly**. Also, **GCE must be enabled**.

	python main.py --algo_name=fm --gce --actors --genres
	python main.py --dataset=ml-1m --gce --actors
	python tune.py --gce --actors --genres
	
`--prepro=<value>`

Defines the pre-processing method used. `<value>` may be `origin`, `Ncore` or `<int>filter` where `<int>` is an integer. Default value is `10filter`.

	python main.py --algo_name=fm --prepro=7filter
	python main.py --dataset=ml-1m --context --prepro=origin

`--num_ng=<value>`

Defines the number of negative samples per positive sample in the training set. `<value>` should be an integer and its default value is 4.

	python main.py --num_ng=10

`--cand_num=<value>`

Number of negative samples in each user test set, where `<value>` is an integer. Default value is 10.

*Relevant to main script:*

`--factors=<value>`

Where `<value>` is an integer with default value 64. Number of latent factors in the model. This parameter can be tuned using `tune.py`.

	python main --algo_name=fm --factors=128
	
`--epochs=<value>`

Where `<value>` is an integer with default value 50. Number of epochs the selected model is trained for. This parameter can be tuned using `tune.py`.

	python main --algo_name=mf --gce --actors --genres --epochs=100
	
`--batch_size=<value>`

Where `<value>` is an integer with default value 256. Training set batch size. This parameter can be tuned using `tune.py`.

`--lr=<value>`

Where `<value>` is a float with default value 0.001. It represents the model's learning rate. This parameter can be tuned using `tune.py`.

*Relevant to tuning:*

`--tune_epochs=<value>`

Defines the number of epochs the tuning script will be run for.

	python tune.py --tune_epochs=20


*Other arguments:*

`--not_early_stopping`: Disables the early stopping mechanism.

`--logsname=<value>`: Sets a specific log file name, where `<value>` is a string.

`--neg_sampling_each_epoch`: Enables generating different negative samplings for each epoch.

`--cut_down_data`: reduces the number of interactions included in the dataset.

`--test_method=loo`: allows to change test splitting method from Time Aware Leave One Out (tloo) to Leave One Out (loo).

`--val_method=loo`: allows to change validation splitting method from tloo to loo.

`--reg_1=<value>` and `--reg_2=<value>`: L1 and L2 regularization, respectively. `<value>` is a float.

`--dropout=<value>`: dropout rate to prevent overfitting. `<value>` is a float.

`--act_func=<value>` and `--out_func=<value>`: define the activation method in interior and output layers respectively. `<value>` should be a string.
