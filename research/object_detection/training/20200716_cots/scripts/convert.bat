call conda activate tensorflow1

set PYTHONPATH=D:\Development\VertigoML\tensorflow_models;D:\Development\VertigoML\tensorflow_models\research;D:\Development\VertigoML\tensorflow_models\research\slim

set TRAINING_PATH=D:\Development\VertigoML\tensorflow_models\research\object_detection\training_sets\20200716_cots
set CONFIG_FILE=%TRAINING_PATH%\config\ssdlite_mobilenet_v2_coco_2018_05_09.config
set CHECKPOINT_PATH=%TRAINING_PATH%\model.ckpt-8284
set OUTPUT_DIR=%TRAINING_PATH%\models

cd D:\Development\VertigoML\tensorflow_models\research\object_detection 

python export_tflite_ssd_graph.py --pipeline_config_path=%CONFIG_FILE% --trained_checkpoint_prefix=%CHECKPOINT_PATH% --output_directory=%OUTPUT_DIR% --add_postprocessing_op=true 

python export_inference_graph.py --pipeline_config_path=%CONFIG_FILE% --trained_checkpoint_prefix=%CHECKPOINT_PATH% --output_directory=%OUTPUT_DIR%  

python tflite_convert.py --graph_def_file=%OUTPUT_DIR%\tflite_graph.pb --output_file=%OUTPUT_DIR%\detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_dev_values=128 --change_concat_input_ranges=false --allow_custom_ops

cd %TRAINING_PATH%\scripts