# Deepstream Facenet

Face Recognition on dGPU and Jetson Nano using DeepStream and Python.

# Docker way:-
`xhost+`
## Jetson
  `docker run -it --rm --net=host --runtime nvidia  -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-5.1 -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/deepstream-l4t:5.1-21.02-samples`
## dGPU
`docker run --gpus all -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-5.1 nvcr.io/nvidia/deepstream:5.1-21.02-triton`

## Overview of App
This demo is built on top of Python sample app [deepstream-test2](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/deepstream-test2) 
 - Download [back-to-back-detectors](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/back-to-back-detectors). It is primary inference i.e PGIE.
                                                                                    or
 - Use FaceDetect TLT [Deployable model from NGC](https://ngc.nvidia.com/catalog/models/nvidia:tlt_facenet/files?version=deployable_v1.0)
 - The secondary inference facenet engine i.e. SGIE
 - Remove or let the tracker be there.
 - Note: embedding dataset (npz file) should be generate by your dataset.
 - Note: you should count avg mean and avg std for your dataset:
    - Put avg mean in offsets parameter and in the net-scale-factor parameter put (1/avg std) in classifier_config.txt to make facenet model work efficient.
 

### Steps to run the demo:

- Generate the engine file for Facenet model
  - facenet_keras.h5 can be found in the models folder. The model is taken from [nyoki-mtl/keras-facenet](https://github.com/nyoki-mtl/keras-facenet)

  - when converting pb file to onnx use below command instead:
  `python -m tf2onnx.convert --input facenet.pb --inputs input_1:0[1,160,160,3] --inputs-as-nchw input_1:0 --outputs Bottleneck_BatchNorm/batchnorm_1/add_1:0 --output facenet.onnx`
  
- Change the model-engine-file path to the your facenet engine file in `classifier_config.txt`.
- `python3 deepstream_test_2.py <h264_elementary_stream_contains_faces>`

## References

Inspired from  https://github.com/riotu-lab/deepstream-facenet
