set PYTHONPATH=D:\Development\VertigoML\tensorflow_models;D:\Development\VertigoML\tensorflow_models\research;D:\Development\VertigoML\tensorflow_models\research\slim

set TRAINING_PATH=D:\Development\VertigoML\tensorflow_models\research\object_detection\training_sets\20200716_cots

set DIR=
cd D:\Development\VertigoML\tensorflow_models\research\object_detection 

python %TRAINING_PATH%\scripts\generate_tfrecord.py --csv_input=%TRAINING_PATH%\via\20200716_all.csv --image_dir=%TRAINING_PATH%\via\20200716_all --output_dir=%TRAINING_PATH%

python legacy/train.py --logtostderr --train_dir=%TRAINING_PATH%/ --pipeline_config_path=%TRAINING_PATH%/config/ssdlite_mobilenet_v2_coco_2018_05_09.config 

cd %TRAINING_PATH%\scripts