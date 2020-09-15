set TFVERSION=tf1
call conda activate %TFVERSION%

rem Possible models
rem faster_rcnn_inception_v2_coco_2018_01_28

set MODEL_TYPE=faster_rcnn_inception_v2_coco_2018_01_28
set MODEL_DIR=D:\Development\VertigoML\tensorflow_models\research\object_detection\training_sets\20200716_cots
set MODEL_PATH=%MODEL_DIR%\models\frozen_inference_graph.pb
set LABEL_PATH=%MODEL_DIR%\config\labelmap.pbtxt

set VIDEO="D:\Datasets\Heron\videos\all-llewellyn_south_west_corner.mp4"
set OUT_VIDEO="D:\Datasets\Heron\videos\all-llewellyn_south_west_corner_processed_0p2.mp4"
set OFFSET=1000

python Object_detection_video.py --model=%MODEL_PATH% --labelmap=%LABEL_PATH% --video=%VIDEO% --subsample=2 --output=%OUT_VIDEO% --offset=0 --threshold=0.2
