NIH sample

1. dataset size: 4000 images.

2. train / test size: 50% / 50%.

3. epochs: 50.

4. training time: 30 minutes; 1 gpu.

5. Algorithm: MobileNet.

   

| noise | training loss | training accuracy | validation loss | validation accuracy | Accuracy |
| ----- | ------------- | ----------------- | --------------- | ------------------- | ----- |
| 0%   | .02          | .81             | .05           | .80                | .73 |
| 10% | .02        | .92          | .08          | .63                | .71 |
| 30%  | .03        | .87             | .10           | .35               | .52 |
| 50%  | .02        | .88              | .09           | .37               | .44 |



## Noise labels selection

1. 20% percent of dataset has been isolated for evaluation.
2. 80% of the dataset for training.
3. the uniqe lables of the dataset saved into array 200 labels.
4. take a random sample from the training dataset and assign random lables from the array.

