NIH sample

1. dataset size: 4000 images.

2. train / test size: 50% / 50%.

3. epochs: 100.

4. training time: 30 hour; 1 gpu.

5. Algorithm: MobileNet

   

| noise | training loss | training accuracy | validation loss | validation accuracy |
| ----- | ------------- | ----------------- | --------------- | ------------------- |
| 0%   | .06           | .95               | .60             | .89                 |
| 5%   | .01           | .99               | .90             | .83                 |
| 10%   | .01          | .99               | .93             | .84                 |
| 30%   | .01          | .99               | .95             | .80                 |
| 50%   | .03           | .98              | 1.49             | .87                 |



## Noise labels selection

1. 30% percent of dataset has been isolated for evaluation
2. 70% of the dataset for training
3. the uniqe lables of the dataset saved into array 200 labels.
4. take a random sample from the training dataset and assign random lables from the array.

