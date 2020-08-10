set PYTHONPATH=D:\Development\VertigoML\tensorflow_models;D:\Development\VertigoML\tensorflow_models\research;D:\Development\VertigoML\tensorflow_models\research\slim

call training_config.bat

set TRAINING_PATH=D:\Development\VertigoML\tensorflow_models\research\object_detection\training_sets\%TRAINING_SET%

set DIR=
cd D:\Development\VertigoML\tensorflow_models\research\object_detection 

mkdir %TRAINING_PATH%\training

python %TRAINING_PATH%\scripts\generate_tfrecord.py --csv_input=%TRAINING_PATH%\via\%DATASET_NAME%.csv --image_dir=%TRAINING_PATH%\via\%DATASET_NAME% --output_dir=%TRAINING_PATH%\training

python legacy/train.py --logtostderr --train_dir=%TRAINING_PATH%\training --pipeline_config_path=%TRAINING_PATH%\config\%CONFIG_NAME%.config 
