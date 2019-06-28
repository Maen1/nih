NIH sample

1. dataset size: 4000 images.
2. epochs: 50.
3. training time: 30 minutes; 1 gpu.
4. Algorithm: MobileNet.
5. train / test size: 70% / 30%.

| noise | training loss | training accuracy | validation loss | validation accuracy | Accuracy |
| ----- | ------------- | ----------------- | --------------- | ------------------- | -------- |
| 0%    | .02           | .90               | .05             | .71                 | .72      |
| 10%   | .02           | .93               | .08             | .56                 | .69      |
| 30%   | .06           | .49               | .10             | .32                 | .56      |
| 50%   | .07           | .35               | .10             | .20                 | .42      |



7. train / test size: 50% / 50%.

| noise | training loss | training accuracy | validation loss | validation accuracy | Accuracy |
| ----- | ------------- | ----------------- | --------------- | ------------------- | ----- |
| 0%   | .02          | .81             | .05           | .80                | .73 |
| 10% | .02        | .92          | .08          | .63                | .71 |
| 30%  | .03        | .87             | .10           | .35               | .52 |
| 50%  | .02        | .88              | .09           | .37               | .44 |



8. train / test size: 30% / 70%.

| noise | training loss | training accuracy | validation loss | validation accuracy | Accuracy |
| ----- | ------------- | ----------------- | --------------- | ------------------- | -------- |
| 0%    | .06           | .89               | .06             | .62                 | .60      |
| 10%   | .02           | .88               | .08             | .54                 | .57      |
| 30%   | .03           | .84               | .09             | .37                 | .48      |
| 50%   | .08           | .40               | .09             | .26                 | .33      |


### Xception

1. train / test size: 30% / 70%.

| noise | training loss | training accuracy | validation loss | validation accuracy | Accuracy |
| ----- | ------------- | ----------------- | --------------- | ------------------- | -------- |
| 0%    | .01           | .94               | .05             | .64                 | .64      |
| 10%   | .01           | .95               | .07             | .56                 | .62     |
| 30%   | .03           | .91               | .07             | .49                 | .51      |
| 50%   | .06		| .45               | .09             | .22                 | .42     |

2. train / test size: 50% / 50%.

| noise | training loss | training accuracy | validation loss | validation accuracy | Accuracy |
| ----- | ------------- | ----------------- | --------------- | ------------------- | -------- |
| 0%    | .01           | .94               | .05             | .66                 | .71      |
| 10%   | .01           | .93               | .08             | .59                 | .67      |
| 30%   | .06           | .60               | .10             | .59                 | .59      |
| 50%   | .02		    | .91               | .08             | .22                 | .46      |

2. train / test size: 70% / 30%.

| noise | training loss | training accuracy | validation loss | validation accuracy | Accuracy |
| ----- | ------------- | ----------------- | --------------- | ------------------- | -------- |
| 0%    | .01           | .96               | .05             | .71                 | .71      |
| 10%   | .01           | .93               | .08             | .52                 | .63      |
| 30%   | .06           | .60               | .10             | .59                 | .59      |
| 50%   | .02		    | .91               | .08             | .22                 | .46      |



## Noise labels selection

1. 20% percent of dataset has been isolated for evaluation.
2. 80% of the dataset for training.
3. the uniqe lables of the dataset saved into array 200 labels.
4. take a random sample from the training dataset and assign random lables from the array.

