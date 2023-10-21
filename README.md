# Scripts and algorithms to use Google's Coral Edge TPU (USB)

## To convert a Tensorflow model:

```
$ python3 src/convert.py /path/to/tf_model.pb /path/to/output.tflite
```
## To compile a TFLite model for the Edge TPU:

```
$ edgetpu_compiler -o /output/folder -s /path/to/tflite_model.tflite
```

## To run an inference with the Edge TPU:

```
$ python3 src/classify_image.py --model /path/to/tflite_model.tflite --labels /path/to/labels --input /path/to/image.jpg
```

