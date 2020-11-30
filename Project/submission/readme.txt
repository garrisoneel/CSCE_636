To run:

train:
set model/train configs in Configs.py
use main.py train <train data dir>
model will be trained and saved using the name set in Config.py

test:
set the model name (must match one of the models saved in saved_models) in Config.py
all other config settings are ignored, so don't worry about them (model is loaded using configs from the saved checkpoint)
use main.py test <train data dir>
model will be evaluated on the test set with the result printed

predict:
set the model name (must match one of the models saved in saved_models) in Config.py
all other config settings are ignored, so don't worry about them (model is loaded using configs from the saved checkpoint)
use main.py predict <private test data dir> --results_dir <results filename>
model will be used to classify the private test set with the result saved to the designated file