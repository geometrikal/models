set TFVERSION=tf1
call conda activate %TFVERSION%

rem Possible models
rem faster_rcnn_inception_v2_coco_2018_01_28

set MODEL_TYPE=faster_rcnn_inception_v2_coco_2018_01_28
set MODEL_DIR="D:\\Development\\VertigoML\\tensorflow_models\\research\\object_detection\\training_sets\\20200716_cots"
set PIPELINE_CONFIG_PATH=%MODEL_DIR%\%TFVERSION%_configs\%MODEL_TYPE%.config
set CHECKPOINT_PATH=%MODEL_DIR%\training\model.ckpt-400000
set OUTPUT_DIR=%MODEL_DIR%\models

python D:\Development\VertigoML\tensorflow_models\research\object_detection/export_inference_graph.py --pipeline_config_path=%PIPELINE_CONFIG_PATH% --trained_checkpoint_prefix=%CHECKPOINT_PATH% --output_directory=%OUTPUT_DIR%