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