NIH sample

1. dataset size: 4000 images.

2. train / test size: 50% / 50%.

3. epochs: 100.

4. training time: 30 hour; 1 gpu.

5. Algorithm: MobileNet

   

| noise | training loss | training accuracy | validation loss | validation accuracy |
| ----- | ------------- | ----------------- | --------------- | ------------------- |
| 0%   | .06           | .95               | .60             | .89                 |
| 10%   | .08          | .93               | .52             | .87                 |
| 30%   | .10          | .92               | .45             | .87                 |
| 50%   | .14           | .90              | .48             | .87                 |



## Noise labels selection

1. 30% percent of dataset has been isolated for evaluation
2. 70% of the dataset for training
3. the uniqe lables of the dataset saved into array 200 labels.
4. take a random sample from the training dataset and assign random lables from the array.

