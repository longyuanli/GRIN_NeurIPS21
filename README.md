# GRIN:  Generative Relation and Intention Network for Multi-agent Trajectory Prediction
Official Implementation of Generative Relation and Intention Network (GRIN) in PyTorch and DGL.

## Dependencies

- torch==1.8.1
- numpy==1.19.2
- scipy==1.6.1
- dgl_cu110==0.6.1
- dgl==0.6.1
- tensorboardX==2.2

## Running the code

0. Install all dependencies mentioned above

1. Generate charged dataset for training (NBA dataset is available on [44])
   
```
python simulator.py --seed 0 --num_sample 5000 --filename train.npz
python simulator.py --seed 1 --num_sample 1000 --filename test.npz
python simulator.py --seed 2 --num_sample 1000 --filename valid.npz
```

2. Train the model

```
bash train.sh
```

3. Evaluate the model

```
bash eval.sh
```