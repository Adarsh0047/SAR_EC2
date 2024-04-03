from mmdet.apis import inference_detector
from mmdet.apis import DetInferencer
import supervision as sv
import numpy as np

# Choose to use a config
model_name = 'tood_r50_fpn_1x_coco.py'
# Setup a checkpoint file to load
checkpoint = 'best_coco_bbox_mAP_epoch_5.pth'

# Set the device to be used for evaluation
device = 'cpu'
model = DetInferencer(model_name, checkpoint, device=device)
# Initialize the DetInferencer
# model = init_detector()

# Use the detector to do inference
# image_path = 
# result = inference_detector(model, image_path)

SOURCE_VIDEO_PATH = "ICEYE Ship.mp4" #Path for the Input tracking video
sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
TARGET_VIDEO_PATH = "ship_result.mp4"# Path to save the predicted (Tracking) Output video.


# create BYTETracker instance
# byte_tracker = sv.ByteTrack(track_thresh=0.1, track_buffer=10, match_thresh=0.1, frame_rate=10)
byte_tracker = sv.ByteTrack()

# create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# create instance of BoxAnnotator
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

# create instance of TraceAnnotator
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=1)

# define call back function to be used in video processing
def callback(frame: np.ndarray, index:int) -> np.ndarray:
    # model prediction on single frame and conversion to supervision Detections
    result = inference_detector(model, frame)
    detections = sv.Detections.from_mmdetection(result)
    detections = detections[detections.confidence > 0.1].with_nms()
    # tracking detections
    detections = byte_tracker.update_with_detections(detections)
    class_id = detections.class_id
    confidence = detections.confidence
    labels = [
        f"{class_id} {confidence:0.2f}"
        for _, _, confidence, class_id,_,_ in detections
    ]
    annotated_frame = trace_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    annotated_frame=box_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels)

    return annotated_frame

# process the whole video and save it to target dir 
sv.process_video(
    source_path = SOURCE_VIDEO_PATH,
    target_path = TARGET_VIDEO_PATH,
    callback=callback
)
