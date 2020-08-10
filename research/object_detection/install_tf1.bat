call conda activate tf1

set TENSORFLOW_MODELS_RESEARCH=D:\Development\VertigoML\tensorflow_models\research
cd %TENSORFLOW_MODELS_RESEARCH%

protoc object_detection/protos/*.proto --python_out=.
copy object_detection\packages\tf1\setup.py .
python -m pip install .
python object_detection/builders/model_builder_tf1_test.py