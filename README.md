# TFG

**Local Execution**

Clone the repository and runn these two lines in terminal from repository's root folder.

```
$ cd project
$ python main.py
```

There are many options available through arguments, which are explained later.

**Remote Execution using Google Colab**

You can also execute the script using Google Colab for better results. Follow the steps in this notebook: https://colab.research.google.com/drive/1Va_Zr-DpotTGD41CHB4uwYu1CEvr5ThV?usp=sharing

The script will run with the default options, you can choose your own arguments by modifying some cells. Please, take care of the original script.

**Arguments**

You can always run this command in order to get the most updated help on arguments:

```
$ python main.py --help
```

--dataset: string. Lets you choose the dataset you want to train the model on.
  Default value: movielens100k
  Available values: movielens100k
  
--add_context: boolean. You can use a context aware system by setting it to True.
  Default value: False
  
--gcn: boolean. Lets you deactivate the Graph Convolutional Network in the Factorization Machine.
  Default value: True
  
--epochs: int. Number of epochs the model will be trained for.
  Default value: 100

--top_k: int. Number of top k recommendations to return.
  Default value: 10

--neg_sample_ratio: int. Number of negative samples generated for every positive sample in the dataset.
  Default value: 4
 
--neg_sample_ratio_test: int. Number of negative samples included in every test set.
  Default value: 99
 
--reduction: 

--batch_size: int. Size of batches generated by our Data Loader
  Default value: 256
  
--device: string. If possible, should be cuda. CPU will always be used if CUDA is not available.
  Default value: cuda if available, cpu if not.
  Available values: cpu, cuda
