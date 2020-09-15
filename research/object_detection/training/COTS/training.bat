@rem WINDOWS VERISON

set TFVERSION=tf1
call conda activate %TFVERSION%

@rem Possible models
@rem faster_rcnn_inception_v2_coco_2018_01_28

set MODEL_TYPE=faster_rcnn_inception_v2_coco_2018_01_28
set PIPELINE_CONFIG_PATH="D:\\Development\\VertigoML\\tensorflow_models\\research\\object_detection\\training_sets\\COTS\\%TFVERSION%_configs\\%MODEL_TYPE%.config"
set MODEL_DIR="D:\\Development\\VertigoML\\tensorflow_models\\research\\object_detection\\training_sets\\COTS\\training"

echo * Model type: %MODEL_TYPE%
echo * Pipeline: %PIPELINE_CONFIG_PATH%
echo * Training directory: %MODEL_DIR%

python D:\Development\VertigoML\tensorflow_models\research\object_detection\model_main.py --pipeline_config_path=%PIPELINE_CONFIG_PATH% --model_dir=%MODEL_DIR% --alsologtostderr