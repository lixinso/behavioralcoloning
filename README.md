# Behavioral Cloning

## Data Collection
1. 3327 records: Data collected from the Beta version Emulator use mouse as angel input
2. 8037 records: from udacity
3. 34703 records: from udacity students sharing

## Data PreProcessing

Resize the data to 66*200*3 to adapt the Nvidia model image size
Combine the all the data source into one input in the data generator and yield it
Split data into train/validation/test
Shuffling
RGB to YUV

## Training, Validation

use fit_generator to generate data and save the memory

The model rarchitecture refer to Nvidia's paper.
![Nvidia Architecture](./source/nvidia_architecture.png)

## Testing

Testing use 10% of the data to do the testing.
Also, pick 1 image from every 1000 images to do the prediction and manully validate the results.

## Improvements

## References
![End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
![Learning a Driving Simulator](https://arxiv.org/pdf/1608.01230v1.pdf)

## TODOs
Add rotation.
Use left and right camera
Add more data to fix the cases when the car off the road


