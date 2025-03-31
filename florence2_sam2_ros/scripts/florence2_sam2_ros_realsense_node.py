#!/your/custom/path/to/python  # Replace this path with the absolute path to your Python interpreter for the Grounded-SAM-2 environment

import os
import cv2
import time
import torch
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import supervision as sv
from PIL import Image as PILImage
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM
from utils.supervision_utils import CUSTOM_COLOR_MAP

"""
Define Some Hyperparam
"""
TASK_PROMPT = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "object_detection": "<OD>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
    "region_proposal": "<REGION_PROPOSAL>",
    "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "region_to_segmentation": "<REGION_TO_SEGMENTATION>",
    "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
    "region_to_category": "<REGION_TO_CATEGORY>",
    "region_to_description": "<REGION_TO_DESCRIPTION>",
    "ocr": "<OCR>",
    "ocr_with_region": "<OCR_WITH_REGION>",
}

# Output directory to save annotated images
OUTPUT_DIR = "./outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

"""
Init Florence-2 and SAM 2 Model
"""
FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
SAM2_CHECKPOINT = rospy.get_param("~sam2_checkpoint", "/your/custom/path/to/sam2.1_hiera_large.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Environment settings
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Set device based on CUDA availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Build Florence-2 model
florence2_model = AutoModelForCausalLM.from_pretrained(
    FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto'
).eval().to(device)
florence2_processor = AutoProcessor.from_pretrained(
    FLORENCE2_MODEL_ID, trust_remote_code=True
)

# Build SAM 2
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam2_predictor = SAM2ImagePredictor(sam2_model)

def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].to(device),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer

def get_images(input_source):
    # Convert file path to PIL and OpenCV images, or pass through numpy array if provided.
    if isinstance(input_source, str):
        pil_image = PILImage.open(input_source).convert("RGB")
        img = cv2.imread(input_source)
    else:
        pil_image = PILImage.fromarray(cv2.cvtColor(input_source, cv2.COLOR_BGR2RGB))
        img = input_source.copy()
    return pil_image, img

"""
Pipeline-1: Object Detection + Segmentation
"""
def object_detection_and_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    input_source,
    task_prompt="<OD>",
    text_input=None,
    output_dir=OUTPUT_DIR
):
    pil_image, img = get_images(input_source)
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, pil_image)

    """ Florence-2 Object Detection Output Format
    {'<OD>': 
        {
            'bboxes': 
                [
                    [33.599998474121094, 159.59999084472656, 596.7999877929688, 371.7599792480469], 
                    [454.0799865722656, 96.23999786376953, 580.7999877929688, 261.8399963378906], 
                    [224.95999145507812, 86.15999603271484, 333.7599792480469, 164.39999389648438], 
                    [449.5999755859375, 276.239990234375, 554.5599975585938, 370.3199768066406], 
                    [91.19999694824219, 280.0799865722656, 198.0800018310547, 370.3199768066406]
                ], 
            'labels': ['car', 'door', 'door', 'wheel', 'wheel']
        }
    }
    """
    results = results[task_prompt]
    # parse florence-2 detection results
    input_boxes = np.array(results["bboxes"])
    class_names = results["labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(pil_image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # specify labels
    labels = [f"{name}" for name in class_names]
    # visualization results
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    
    if isinstance(input_source, str):
        cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_det_annotated_image.jpg"), annotated_frame)
        cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_det_image_with_mask.jpg"), annotated_frame)
        rospy.loginfo('Successfully saved annotated image to "{}"'.format(output_dir))
    
    return annotated_frame

"""
Pipeline 2: Dense Region Caption + Segmentation
"""
def dense_region_caption_and_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    input_source,
    task_prompt="<DENSE_REGION_CAPTION>",
    text_input=None,
    output_dir=OUTPUT_DIR
):
    assert text_input is None, "Text input should be None when calling dense region caption pipeline."
    pil_image, img = get_images(input_source)
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, pil_image)

    """ Florence-2 Object Detection Output Format
    {'<DENSE_REGION_CAPTION>': 
        {
            'bboxes': 
                [
                    [33.599998474121094, 159.59999084472656, 596.7999877929688, 371.7599792480469], 
                    [454.0799865722656, 96.23999786376953, 580.7999877929688, 261.8399963378906], 
                    [224.95999145507812, 86.15999603271484, 333.7599792480469, 164.39999389648438], 
                    [449.5999755859375, 276.239990234375, 554.5599975585938, 370.3199768066406], 
                    [91.19999694824219, 280.0799865722656, 198.0800018310547, 370.3199768066406]
                ], 
            'labels': ['turquoise Volkswagen Beetle', 'wooden double doors with metal handles', 'wheel', 'wheel', 'door']
        }
    }
    """
    results = results[task_prompt]
    # parse florence-2 detection results
    input_boxes = np.array(results["bboxes"])
    class_names = results["labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(pil_image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # specify labels
    labels = [f"{name}" for name in class_names]
    # visualization results
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    
    if isinstance(input_source, str):
        cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_dense_region_cap_annotated_image.jpg"), annotated_frame)
        cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_dense_region_cap_image_with_mask.jpg"), annotated_frame)
        print(f'Successfully saved annotated image to "{output_dir}"')
    
    return annotated_frame

"""
Pipeline 3: Region Proposal + Segmentation
"""
def region_proposal_and_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    input_source,
    task_prompt="<REGION_PROPOSAL>",
    text_input=None,
    output_dir=OUTPUT_DIR
):
    assert text_input is None, "Text input should be None when calling region proposal pipeline."
    pil_image, img = get_images(input_source)
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, pil_image)

    """ Florence-2 Object Detection Output Format
    {'<REGION_PROPOSAL>': 
        {
            'bboxes': 
                [
                    [33.599998474121094, 159.59999084472656, 596.7999877929688, 371.7599792480469], 
                    [454.0799865722656, 96.23999786376953, 580.7999877929688, 261.8399963378906], 
                    [224.95999145507812, 86.15999603271484, 333.7599792480469, 164.39999389648438], 
                    [449.5999755859375, 276.239990234375, 554.5599975585938, 370.3199768066406], 
                    [91.19999694824219, 280.0799865722656, 198.0800018310547, 370.3199768066406]
                ], 
            'labels': ['', '', '', '', '', '', '']
        }
    }
    """
    results = results[task_prompt]
    # parse florence-2 detection results
    input_boxes = np.array(results["bboxes"])
    class_names = results["labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(pil_image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # specify labels
    labels = [f"region_{idx}" for idx, _ in enumerate(class_names)]
    # visualization results
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    
    if isinstance(input_source, str):
        cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_region_proposal.jpg"), annotated_frame)
        cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_region_proposal_with_mask.jpg"), annotated_frame)
        print(f'Successfully saved annotated image to "{output_dir}"')
    
    return annotated_frame

"""
Pipeline 4: Phrase Grounding + Segmentation
"""
def phrase_grounding_and_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    input_source,
    task_prompt="<CAPTION_TO_PHRASE_GROUNDING>",
    text_input=None,
    output_dir=OUTPUT_DIR
):
    assert text_input is not None, "Text input should not be None when calling phrase grounding pipeline."
    pil_image, img = get_images(input_source)
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, pil_image)

    """ Florence-2 Object Detection Output Format
    {'<CAPTION_TO_PHRASE_GROUNDING>': 
        {
            'bboxes': 
                [
                    [34.23999786376953, 159.1199951171875, 582.0800170898438, 374.6399841308594], 
                    [1.5999999046325684, 4.079999923706055, 639.0399780273438, 305.03997802734375]
                ], 
            'labels': ['A green car', 'a yellow building']
        }
    }
    """
    results = results[task_prompt]
    # parse florence-2 detection results
    input_boxes = np.array(results["bboxes"])
    class_names = results["labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(pil_image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # specify labels
    labels = [f"{name}" for name in class_names]
    # visualization results
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    
    if isinstance(input_source, str):
        cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_phrase_grounding.jpg"), annotated_frame)
        cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_phrase_grounding_with_mask.jpg"), annotated_frame)
        print(f'Successfully saved annotated image to "{output_dir}"')
    
    return annotated_frame

"""
Pipeline 5: Referring Expression Segmentation

Note that Florence-2 directly support referring segmentation with polygon output format, which may be not that accurate, 
therefore we try to decode box from polygon and use SAM 2 for mask prediction
"""
def referring_expression_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    input_source,
    task_prompt="<REFERRING_EXPRESSION_SEGMENTATION>",
    text_input=None,
    output_dir=OUTPUT_DIR
):
    assert text_input is not None, "Text input should not be None when calling referring segmentation pipeline."
    pil_image, img = get_images(input_source)
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, pil_image)

    """ Florence-2 Object Detection Output Format
    {'<REFERRING_EXPRESSION_SEGMENTATION>': 
        {
            'polygons': [[[...]]]
            'labels': ['']
        }
    }
    """
    results = results[task_prompt]
    
    # parse florence-2 detection results
    polygon_points = np.array(results["polygons"][0], dtype=np.int32).reshape(-1, 2)
    class_names = [text_input]
    class_ids = np.array(list(range(len(class_names))))
    
    # parse polygon format to mask
    img_width, img_height = pil_image.size
    florence2_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    if len(polygon_points) < 3:
        print("Invalid polygon:", polygon_points)
        exit()
    cv2.fillPoly(florence2_mask, [polygon_points], 1)
    if florence2_mask.ndim == 2:
        florence2_mask = florence2_mask[None]
    
    # compute bounding box based on polygon points
    x_min = np.min(polygon_points[:, 0])
    y_min = np.min(polygon_points[:, 1])
    x_max = np.max(polygon_points[:, 0])
    y_max = np.max(polygon_points[:, 1])

    input_boxes = np.array([[x_min, y_min, x_max, y_max]])
    
    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(pil_image))
    sam2_masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    if sam2_masks.ndim == 4:
        sam2_masks = sam2_masks.squeeze(1)
    
    # specify labels
    labels = [f"{text_input}"]
    # visualization florence2 mask
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=florence2_mask.astype(bool),
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    
    if isinstance(input_source, str):
        cv2.imwrite(os.path.join(output_dir, "florence2_referring_segmentation_box.jpg"), annotated_frame)
        cv2.imwrite(os.path.join(output_dir, "florence2_referring_segmentation_box_with_mask.jpg"), annotated_frame)
        print(f'Successfully saved florence-2 annotated image to "{output_dir}"')
    
    # visualize sam2 mask
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=sam2_masks.astype(bool),
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
  
    if isinstance(input_source, str):
        cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_referring_box.jpg"), annotated_frame)
        cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_referring_box_with_sam2_mask.jpg"), annotated_frame)
        print(f'Successfully saved sam2 annotated image to "{output_dir}"')
    
    return annotated_frame

"""
Pipeline 6: Open-Vocabulary Detection + Segmentation
"""
def open_vocabulary_detection_and_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    input_source,
    task_prompt="<OPEN_VOCABULARY_DETECTION>",
    text_input=None,
    output_dir=OUTPUT_DIR
):
    assert text_input is not None, "Text input should not be None when calling open-vocabulary detection pipeline."
    pil_image, img = get_images(input_source)
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, pil_image)
    
    """ Florence-2 Open-Vocabulary Detection Output Format
    {'<OPEN_VOCABULARY_DETECTION>': 
        {
            'bboxes': 
                [
                    [34.23999786376953, 159.1199951171875, 582.0800170898438, 374.6399841308594]
                ], 
            'bboxes_labels': ['A green car'],
            'polygons': [], 
            'polygons_labels': []
        }
    }
    """
    results = results[task_prompt]
    
    # parse florence-2 detection results
    input_boxes = np.array(results["bboxes"])
    print(results)
    class_names = results["bboxes_labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(pil_image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # specify labels
    labels = [f"{name}" for name in class_names]
    # visualization results
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    
    if isinstance(input_source, str):
        cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_open_vocabulary_detection.jpg"), annotated_frame)
        cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_open_vocabulary_detection_with_mask.jpg"), annotated_frame)
        print(f'Successfully saved annotated image to "{output_dir}"')
    
    return annotated_frame


class GroundedSAM2ROSNode:
    def __init__(self):
        rospy.init_node('grounded_sam2_realsense_ros_node', anonymous=True)
        self.bridge = CvBridge()
        
        # Retrieve the 'pipeline' parameter to select processing pipeline
        self.pipeline = rospy.get_param("~pipeline", "object_detection_segmentation")
        self.text_input = rospy.get_param("~text_input", None)
        
        # Use the pre-loaded models
        self.florence2_model = florence2_model
        self.florence2_processor = florence2_processor
        self.sam2_predictor = sam2_predictor
        
        # Subscribe to RealSense image topic (ex: /camera/color/image_raw)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1)
        # Publisher for annotated images
        self.image_pub = rospy.Publisher("/grounded_sam2/segmented_image", Image, queue_size=1)
        
        rospy.loginfo("GroundedSAM2ROSNode initialized. Subscribed to /camera/color/image_raw")
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("Error converting ROS image to OpenCV: {}".format(e))
            return
        
        # Define pipeline function mapping
        pipeline_functions = {
            "object_detection_segmentation": object_detection_and_segmentation,
            "dense_region_caption_segmentation": dense_region_caption_and_segmentation,
            "region_proposal_segmentation": region_proposal_and_segmentation,
            "phrase_grounding_segmentation": phrase_grounding_and_segmentation,
            "referring_expression_segmentation": referring_expression_segmentation,
            "open_vocabulary_detection_segmentation": open_vocabulary_detection_and_segmentation,
        }
        
        if self.pipeline not in pipeline_functions:
            rospy.logerr("Unsupported pipeline: {}".format(self.pipeline))
            return

        try:
            # Process the image using the selected pipeline function
            annotated_frame = pipeline_functions[self.pipeline](
                florence2_model=self.florence2_model,
                florence2_processor=self.florence2_processor,
                sam2_predictor=self.sam2_predictor,
                input_source=cv_image,
                text_input=self.text_input
            )
        except Exception as e:
            rospy.logerr("Error in processing pipeline {}: {}".format(self.pipeline, e))
            return
        
        try:
            # Convert annotated OpenCV image back to ROS Image message and publish.
            result_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.image_pub.publish(result_msg)
        except Exception as e:
            rospy.logerr("Error converting annotated image to ROS image: {}".format(e))


if __name__ == "__main__":
    try:
        node = GroundedSAM2ROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
