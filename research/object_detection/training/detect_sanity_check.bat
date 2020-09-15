set BASE=D:\Development\VertigoML\tensorflow_models\research\object_detection\training_sets\sanity_check

set IMAGE=%BASE%\pmam.jpg

python TFLite_detection_image.py --modeldir %BASE% --labels %BASE%\labelmap.txt --threshold 0.1 --image %IMAGE%


REM parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
REM                     required=True)
REM parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
REM                     default='detect.tflite')
REM parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
REM                     default='labelmap.txt')
REM parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
REM                     default=0.5)
REM parser.add_argument('--image', help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir',
REM                     default=None)
REM parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
REM                     default=None)
REM parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
REM                     action='store_true')