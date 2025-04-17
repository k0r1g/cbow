## Cbow


```sh
$ mkdir -p ~/miniconda3
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ rm ~/miniconda3/miniconda.sh
$ source ~/miniconda3/bin/activate
$ conda init --all
```

```sh
$ conda create --name wrd python=3.11 -y
$ conda activate wrd
$ pip install torch wandb
```


```sh
$ python 00_train_tkn.py
$ python 01_train_w2v.py
$ python 02_train_reg.py
```
