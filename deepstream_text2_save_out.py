#Import and Initialize
import sys
sys.path.append('../')
import configparser
import numpy as np
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst, GstRtspServer#
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS

from facenet_utils import load_dataset, normalize_vectors, predict_using_classifier

import ctypes
import pyds
import time

fps_stream=None
face_counter= []

PGIE_CLASS_ID_PERSON = 2
SGIE_CLASS_ID_FACE = 0

PRIMARY_DETECTOR_UID = 1
SECONDARY_DETECTOR_UID = 2

#face embeddings
DATASET_PATH = 'embeddings/my_embed.npz'

faces_embeddings, labels = load_dataset(DATASET_PATH)

# Defining the input output video file 
stream_path  = '/opt/nvidia/deepstream/deepstream-5.1/samples/streams/sample_720p.h264'
output_path = "ds_out.mp4"

fps_stream = GETFPS(0)
print(fps_stream)
# Standard GStreamer initialization
GObject.threads_init()
Gst.init(None)

# Create gstreamer elements
# Create Pipeline element that will form a connection of other elements
print("Creating Pipeline \n ")
pipeline = Gst.Pipeline()

if not pipeline:
    sys.stderr.write(" Unable to create Pipeline \n")
    
## Make Element or Print Error and any other detail
def make_elm_or_print_err(factoryname, name, printedname, detail=""):
  print("Creating", printedname)
  elm = Gst.ElementFactory.make(factoryname, name)
  if not elm:
     sys.stderr.write("Unable to create " + printedname + " \n")
  if detail:
     sys.stderr.write(detail)
  return elm
  
  
# Source element for reading from the file
source = make_elm_or_print_err("filesrc", "file-source","Source")
# Since the data format in the input file is elementary h264 stream,
# we need a h264parsermake_elm_or_print_err("filesrc", "file-source","Source")
h264parser = make_elm_or_print_err("h264parse", "h264-parser", "h264 parser")
# Use nvdec_h264 for hardware accelerated decode on GPU
decoder = make_elm_or_print_err("nvv4l2decoder", "nvv4l2-decoder", "Decoder")
# Create nvstreammux instance to form batches from one or more sources.
streammux = make_elm_or_print_err("nvstreammux", "Stream-muxer", "NvStreamMux")
# Use nvinfer to run inferencing on decoder's output,
# behaviour of inferencing is set through config file
#Detector
face_detector = make_elm_or_print_err("nvinfer", "primary-inference face detector", "face_detector")
#Classifier
face_classifier = make_elm_or_print_err("nvinfer", "secondary-inference face_classifier", "face_classifier")
# Use convertor to convert from NV12 to RGBA as required by nvosd
nvvidconv = make_elm_or_print_err("nvvideoconvert", "convertor", "nvvidconv")
# Create OSD to draw on the converted RGBA buffer
nvosd = make_elm_or_print_err("nvdsosd", "onscreendisplay", "nvosd" )
# Finally encode and save the osd output
queue = make_elm_or_print_err("queue", "queue", "Queue")
# Use convertor to convert from NV12 to RGBA as required by nvosd
nvvidconv2 = make_elm_or_print_err("nvvideoconvert", "convertor2","nvvidconv2")
# Place an encoder instead of OSD to save as video file
encoder = make_elm_or_print_err("avenc_mpeg4", "encoder", "Encoder")
# Parse output from Encoder 
codeparser = make_elm_or_print_err("mpeg4videoparse", "mpeg4-parser", 'Code Parser')
# Create a container
container = make_elm_or_print_err("qtmux", "qtmux", "Container")
# Create Sink for storing the output 
sink = make_elm_or_print_err("filesink", "filesink", "Sink")


############ Set properties for the Elements ############
print("Playing file %s", stream_path)
# Set Input File Name 
source.set_property('location', stream_path)
# Set Input Width , Height and Batch Size 
streammux.set_property('width', 1920)
streammux.set_property('height', 1080)
streammux.set_property('batch-size', 1)

# Set Timeout in microseconds to wait after the first buffer is  
# available to push the batch even if a complete batch is not formed.
streammux.set_property('batched-push-timeout', 4000000)

# Set Congifuration file for nvinfer 
face_detector.set_property('config-file-path', "detector_config.txt")
face_classifier.set_property('config-file-path', "classifier_config.txt")

# Set Encoder bitrate for output video
encoder.set_property("bitrate", 2000000)
sink.set_property('location', output_path)
sink.set_property('async', False)
sink.set_property('sync', 1)


print("Adding elements to Pipeline \n")
pipeline.add(source)
pipeline.add(h264parser)
pipeline.add(decoder)
pipeline.add(streammux)
pipeline.add(face_detector)
pipeline.add(face_classifier)
pipeline.add(nvvidconv)
pipeline.add(nvosd)
pipeline.add(queue)
pipeline.add(nvvidconv2)
pipeline.add(encoder)
pipeline.add(codeparser)
pipeline.add(container)
pipeline.add(sink)

if is_aarch64():
    pipeline.add(transform)

# we link the elements together
# file-source -> h264-parser -> nvh264-decoder ->
# nvinfer -> nvvidconv -> nvosd -> video-renderer
print("Linking elements in the Pipeline \n")
source.link(h264parser)
h264parser.link(decoder)

sinkpad = streammux.get_request_pad("sink_0")
if not sinkpad:
    sys.stderr.write(" Unable to get the sink pad of streammux \n")
srcpad = decoder.get_static_pad("src")
if not srcpad:
    sys.stderr.write(" Unable to get source pad of decoder \n")

srcpad.link(sinkpad)
streammux.link(face_detector)
face_detector.link(face_classifier)
face_classifier.link(nvvidconv)
nvvidconv.link(nvosd)

nvosd.link(queue)
queue.link(nvvidconv2)
nvvidconv2.link(encoder)
encoder.link(codeparser)
codeparser.link(container)
container.link(sink)

loop = GObject.MainLoop()

bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect ("message", bus_call, loop)


def osd_sink_pad_buffer_probe(pad,info,u_data):
    global fps_stream, face_counter
    frame_number=0
    #Intiallizing object counter with 0.
    person_count = 0
    face_count = 0
    num_rects=0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.glist_get_nvds_frame_meta()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list

        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            if obj_meta.unique_component_id == PRIMARY_DETECTOR_UID:
                if obj_meta.class_id == PGIE_CLASS_ID_PERSON:
                   person_count += 1
                
            if obj_meta.unique_component_id == SECONDARY_DETECTOR_UID:
                if obj_meta.class_id == SGIE_CLASS_ID_FACE:
                   face_count += 1
            try:
                l_obj=l_obj.next
            except StopIteration:
                break
        fps_stream.get_fps()
        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Face Count={}".format(frame_number, num_rects, num_rects)
        face_counter.append(person_count)

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK
    
    
# Lets add probe to get informed of the meta data generated, we add probe to
# the sink pad of the osd element, since by that time, the buffer would have
# had got all the metadata.
osdsinkpad = nvosd.get_static_pad("sink")
if not osdsinkpad:
    sys.stderr.write(" Unable to get sink pad of nvosd \n")

osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)


def sgie_sink_pad_buffer_probe(pad,info,u_data):

    frame_number=0

    num_rects=0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    #print("BM:", batch_meta)
    l_frame = batch_meta.frame_meta_list
    #print("l_frame:", l_frame)
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta

        l_obj=frame_meta.obj_meta_list
        #print("l_obj:", l_obj)
        
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            #print("obj_meta:",obj_meta)  
            l_user = obj_meta.obj_user_meta_list
            while l_user is not None:

                try:
                    # Casting l_user.data to pyds.NvDsUserMeta
                    user_meta=pyds.NvDsUserMeta.cast(l_user.data)
                except StopIteration:
                    break

                if (
                    user_meta.base_meta.meta_type
                    != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
                ):
                    continue

                # Converting to tensor metadata
                # Casting user_meta.user_meta_data to NvDsInferTensorMeta
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

                # Get output layer as NvDsInferLayerInfo
                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)

                # Convert NvDsInferLayerInfo buffer to numpy array
                ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                v = np.ctypeslib.as_array(ptr, shape=(128,))

                # Pridict face neme
                yhat = v.reshape((1,-1))
                face_to_predict_embedding = normalize_vectors(yhat)
                result = predict_using_classifier(faces_embeddings, labels, face_to_predict_embedding)
                result =  (str(result).title())
                print('Predicted name: %s' % result)
                # Generate classifer metadata and attach to obj_meta
                # Get NvDsClassifierMeta object
                classifier_meta = pyds.nvds_acquire_classifier_meta_from_pool(batch_meta)

                # Pobulate classifier_meta data with pridction result
                classifier_meta.unique_component_id = tensor_meta.unique_id


                label_info = pyds.nvds_acquire_label_info_meta_from_pool(batch_meta)


                label_info.result_prob = 0
                label_info.result_class_id = 0

                pyds.nvds_add_label_info_meta_to_classifier(classifier_meta, label_info)
                pyds.nvds_add_classifier_meta_to_object(obj_meta, classifier_meta)

                display_text = pyds.get_string(obj_meta.text_params.display_text)
                obj_meta.text_params.display_text = f'{display_text} {result}'

                try:
                    l_user = l_user.next
                except StopIteration:
                    break

            try:
                l_obj=l_obj.next
            except StopIteration:
                break
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


vidconvsinkpad = nvvidconv.get_static_pad("sink")

if not vidconvsinkpad:
    sys.stderr.write(" Unable to get sink pad of nvvidconv \n")

vidconvsinkpad.add_probe(Gst.PadProbeType.BUFFER, sgie_sink_pad_buffer_probe, 0)


# start play back and listen to events
print("Starting pipeline \n")
start_time = time.time()
pipeline.set_state(Gst.State.PLAYING)
try:
    loop.run()
except:
    pass
# cleanup
pipeline.set_state(Gst.State.NULL)
print("--- %s seconds ---" % (time.time() - start_time))
