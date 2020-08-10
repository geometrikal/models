call training_config.bat

set OUTPUT_DIR=%TRAINING_PATH%\models
set CONFIG_DIR=%TRAINING_PATH%\config
set IMAGE_DIR=%TRAINING_PATH%\test_images

python TFLite_detection_image.py --modeldir %OUTPUT_DIR% --labels %CONFIG_DIR%\labelmap.txt --threshold 0.5 --imagedir %IMAGE_DIR% --subsample 2

rem --edgetpu
