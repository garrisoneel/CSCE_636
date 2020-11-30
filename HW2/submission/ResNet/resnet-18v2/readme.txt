	### YOUR CODE HERE
	flags.DEFINE_integer('resnet_version', 2, 'the version of ResNet')
	flags.DEFINE_integer('resnet_size', 18, 'n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
	flags.DEFINE_integer('batch_size', 128, 'training batch size')
	flags.DEFINE_integer('num_classes', 10, 'number of classes')
	flags.DEFINE_integer('save_interval', 10, 'save the checkpoint when epoch MOD save_interval == 0')
	flags.DEFINE_integer('first_num_filters', 16, 'number of classes')
	flags.DEFINE_float('weight_decay', 2e-4, 'weight decay rate')
	flags.DEFINE_string('modeldir', 'resnet-18v2', 'model directory')
	### END CODE HERE