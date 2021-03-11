# DeepFake Detection Using ScatterNets



## Installation

To run this repository it is highly recommended to create a separate anaconda enviroment. 

Create a anaconda enviroment:
```
conda create --name deepfake_detection
```

Activate the created requirement:
```
conda activate deepfake_detection
```

Run the requiremnts file to get the requirements:
```
pip install -r requirements.txt
```

This repo uses my pytorch implementation of the dtcwt: [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets). You can install this however just by pip installing the requirements.txt. From the command line, the following 3 commands will allow you to run the experiments:
```
git clone https://github.com/fbcotter/scatnet_learn
pip install -r requirements.txt
pip install .
```

## Dataset

To download the datasets please follow the folowwing [repository](https://github.com/Qingcsai/awesome-Deepfakes#2-deepfakes-datasets). 

Preprocessing has to be performed on the dataset with an appropriate number of frames sampled per second. 
The code for this will be updated shortly. 

## Training the model

The minimum required argument are the following.
```
python train.py --model_name "ScatterNetDeepFakeDetection" --dataset "FaceForensics"

```

The complete list of argumenta that can be provided is as follows:

```
model_name: (str) Write the name of your model.
dataset: (str) The deepfake detection to train your model on. 
            Availabel choices are: 'FaceForensics', 'FaceForensics++', 'CelebDF', 'GoogleDFD'\
        ,'FaceHQ', 'DFDC', 'DeeperForensics', 'UADFV'

num_epochs: (int) Enter the number of epochs to train the network.

loss_type: (str) Enter the desired loss type.

scheduler_type: (str) Enter the scheduler type to vary the learning rate.

log_interval: (int) Enter the intervals after which to log the details.

num_classes: (int) Enter the number of classes for a given dataset.

batch_size: (int) Enter the number of batch size.

optimizer: (str) Enter the optimizer required to step during training.

lr: (float) Enter the learning rate to train the network.

augmentation: (bool) Augmentations to the training set to make the network more generalizable.

save_path: (str) Save path for models.

logdir: (str) Log dir for tensorboard.

flush_history: (bool) Flush the tensorboard log dir.

load_model: (bool) Start training the model from a checkpoint.

parallel: (bool) Activate if multiple gpus used.

val: (bool) Activate to use validation set.

return_best: (bool) Returns the best model after training.

```

## Testing the model

Run the following command to test the model

```
python test.py --model_name "ScatterNetDeepFakeDetection" --dataset "FaceForensics" --frames 30

```
