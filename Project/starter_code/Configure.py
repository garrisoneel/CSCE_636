# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE
from Network import StandardBlock, ImprovedBlock, BottleneckBlock
model_configs = {
	"name": 'resnet-18-improved-erasing',
	"save_dir": '../saved_models/',
	"depth": 18,
	"block": ImprovedBlock,
	"num_classes": 10
	# ...
}

training_configs = {
	"learning_rate": 0.1,
	"batch_size": 128,
	"momentum": 0.9,
	"save_interval": 10,
	"weight_decay": 2e-4, 
	"max_epoch": 350,
	"patience": 5,
	"early_stop": 30,
	"lr_schedule": {100:0.1, 200:0.01, 350:0.001}
	# ...
}

### END CODE HERE