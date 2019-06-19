NIH sample

1. dataset size: 4000 images.

2. train / test size: 70% / 30%.

3. epochs: 100.

4. training time: 1 hour; 1 gpus.

5. Algorithm: Xception

   

| noise | training loss | training accuracy | validation loss | validation accuracy |
| ----- | ------------- | ----------------- | --------------- | ------------------- |
| 10%   | .29           | .89               | .32             | .89                 |
| 50%   | .06           | .95               | .63             | .86                 |
| 70%   | .18           | .92               | .42             | .87                 |



## Noise labels selection

1. 30% percent of dataset has been isolated for evaluation
2. 70% of the dataset for training
3. the uniqe lables of the dataset saved into array 200 labels.
4. take a random sample from the training dataset and assign random lables from the array.