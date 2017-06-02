# naive_network_embedding
Naive network embedding using skip gram model, developed by pytorch

#Install
To install the dependency library, you need run `pip install -r requirments.txt`. It will install tqdm, a process bar for python when running in terminal.

#Run
To run the model, you need to run `python main.py datafile`, where `datafile` is the dataset file.
I upload two datasets, karate and zhihu. So if you want use zhihu, you can run with `python main.py zhihu.edgelist`

#Problems
I have the following problems:
* The training speed is too slow. The tqdm will show the running speed. On my MacBook Pro, it can train 100 edges per second. Note that this include the positive edge and 20 negative edges. However, the same model implemented by DyNet could handle 10k edges per second.
* The training speed will decrease with time when use zhihu dataset. The initial speed could be 100 edges per second. But after one minute, the speed will be only 20 edges per second.
