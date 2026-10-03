
#Set hyper-parameters for our training of neural networks
LEARNING_RATE = 0.001
EPOCH = 100
VERBOSE = 1

TRAINING_SIZE = 3000
TEST_SIZE = 2000

#No of target models
NUM_TARGET = 1
#No of shadow models
NUM_SHADOW = 6

#Label value "in" for records present in training data of shadow models
IN = 1
#Label value "out" for records not present in training data of shadow models
OUT = 0