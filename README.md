## Peak counter

## Tensor flow required

$ source ~/tensorflow/bin/activate

## Demo for 10 peaks counter

Run the following in terminal to train
```
$ python training.py --train simdata/Training.csv --num_epochs 20 --verbose True
```

Run the following in terminal to test
```
$ python estimate_output.py --test simdata/Test.csv --verbose True
```
