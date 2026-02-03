# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# hf login --token YOUR_HF_TOKEN
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import AutoModelForCausalLM

load_dotenv(".env")
login(os.getenv("HF_TOKEN"))

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


class MoonDreamHelper:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_loaded = False
        self.latencies = []
        self.detection_class = kwargs.get("detection_class", "door")
        self.query = kwargs.get(
            "query",
            "Is there a door in the image? Ignore vehicle doors. Answer yes or no.",
        )
        self.query_only = kwargs.get("query_only", True)
        self.reasoning = kwargs.get("reasoning", False)
        if isinstance(self.detection_class, str):
            self.detection_class = [self.detection_class]
        if isinstance(self.query, str):
            self.query = [self.query]
        self.default_query = [
            f"Is there a {cls} in the image? Answer yes or no."
            for cls in self.detection_class
        ]

    def load_model(self):
        moondream = AutoModelForCausalLM.from_pretrained(
            "moondream/moondream3-preview",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map={"": "cuda"},
        )
        moondream.compile()
        self.model = moondream
        self.model_loaded = True

    def infer_image(self, img_path, threshold=0.001):
        # return bboxes, scores, classes
        t0 = time.perf_counter()
        image = Image.open(img_path)
        result = self.model.detect(image, self.detection_class)
        self.latencies.append(time.perf_counter() - t0)
        # change dets xyxy to xywh
        if "objects" not in result:
            print(f"No objects detected in {Path(img_path).name}")
            return 0
        bboxes = []
        image_width, image_height = image.size
        for bbox in result["objects"]:
            # Convert normalized coordinates (0-1) to pixel coordinates
            x_min = int(bbox["x_min"] * image_width)
            y_min = int(bbox["y_min"] * image_height)
            x_max = int(bbox["x_max"] * image_width)
            y_max = int(bbox["y_max"] * image_height)
            bboxes.append([x_min, y_min, x_max - x_min, y_max - y_min])

        klass_ids = [1] * len(bboxes)  # all are class 1
        confidences = [1.0] * len(bboxes)
        return bboxes, confidences, klass_ids

    def get_model(self):
        return self.model

    def detect_object(self, encoded_image, detection_classes):
        """Process a single image and save the overlayed result."""
        detection_results = []
        for i, detection_class in enumerate(detection_classes):
            result = self.model.detect(encoded_image, detection_class)
            for bbox in result["objects"]:
                detection_results.append(
                    [
                        i,
                        bbox["x_min"],
                        bbox["y_min"],
                        bbox["x_max"] - bbox["x_min"],
                        bbox["y_max"] - bbox["y_min"],
                    ]
                )

        return detection_results

    def query_image(self, encoded_image, query):
        # run with reasoning and without reasoning
        result = self.model.query(image=encoded_image, question=query, reasoning=False)
        ans_without_reasoning = result["answer"].lower()
        result = self.model.query(image=encoded_image, question=query, reasoning=True)
        ans_with_reasoning = result["answer"].lower()
        if "yes" in ans_with_reasoning and "yes" in ans_without_reasoning:
            return True
        elif "no" in ans_with_reasoning or "no" in ans_without_reasoning:
            return False
        else:
            return None

    def advanced_query_detect(self, image_path, queries=[], detection_classes=[]):
        """Query and detect objects in an image."""
        image_path = Path(image_path)
        image = Image.open(image_path)
        encoded_image = self.model.encode_image(image)
        query_results = {}
        infer_detections = True

        for query in queries:
            query_results[query] = self.query_image(encoded_image, query=query)

        if not all(query_results.values()):
            return query_results, []

        detection_results = self.detect_object(
            encoded_image, detection_classes=detection_classes
        )

        return query_results, detection_results

    def process_folder(self, input_dir, output_dir):
        """Process all images in a folder and save overlayed results to output folder."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        predictions_dir = output_dir / "pred_txt"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        if not input_dir.exists():
            raise ValueError(f"Input directory {input_dir} does not exist")
        if not self.model_loaded:
            self.load_model()
        # Get all image files
        image_files = []
        for ext in IMG_EXTENSIONS:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

        if not image_files:
            print(f"No image files found in {input_dir}")
            return

        print(f"Found {len(image_files)} images to process")

        total_detections = 0
        total_images_with_doors = 0
        images_has_doors = {}
        for image_file in tqdm(image_files, desc="Processing images"):
            # if image_file.stem != 'glass_doors_00103':
            #     continue
            try:
                query_results, detections = self.advanced_query_detect(
                    image_file,
                    # queries=[f"Is there a {cls} in the image? Answer yes or no." for cls in self.detection_class],
                    queries=self.query if self.query else self.default_query,
                    detection_classes=self.detection_class,
                )
                images_has_doors[image_file.stem] = query_results
                if len(detections) == 0:
                    continue
                # save detections as txt
                detection_txt_path = predictions_dir / f"{image_file.stem}.txt"
                with open(detection_txt_path, "w") as f:
                    for det in detections:
                        cls_id, x_min, y_min, width, height = det
                        f.write(f"{cls_id} {x_min} {y_min} {width} {height}\n")
                total_detections += len(detections)

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        print(f"Processing complete! Total detections: {total_detections}")

        if len(images_has_doors) > 0:
            print(
                f"Total images with '{self.detection_class}': {total_images_with_doors} out of {len(image_files)}"
            )
            # save results to a json
            # reasoning_tag = "with_reasoning" if self.reasoning else "no_reasoning"
            reasoning_tag = "advanced"
            results_path = output_dir / f"query_results_{reasoning_tag}.json"
            with open(results_path, "w") as f:
                json.dump(images_has_doors, f, indent=4)
            print(f"Query results saved to {results_path}")


def difference_in_query_results(json_path1, json_path2):
    """Compare two JSON files with query results and print differences."""
    with open(json_path1, "r") as f:
        results1 = json.load(f)
    with open(json_path2, "r") as f:
        results2 = json.load(f)

    differing_images = []
    for image_name in results1:
        if image_name in results2:
            if results1[image_name] != results2[image_name]:
                differing_images.append(image_name)

    print(f"Total differing images: {len(differing_images)}")
    for img in differing_images:
        print(
            f"- {input_folder / img}: {results1[img]} (json1) vs {results2[img]} (json2)"
        )


def print_false_negatives(json_path):
    """Print images where the model predicted 'no' but there is actually an object."""
    with open(json_path, "r") as f:
        results = json.load(f)

    false_negatives = [img for img, has_object in results.items() if not has_object]
    print(f"Total false negatives: {len(false_negatives)}")
    for img in false_negatives:
        print(f"- {input_folder / img}")
    print(f"Total positives: {len(results) - len(false_negatives)}/{len(results)}")


def merge_annotations(txt_folder, output_txt_folder):
    """Merge annotations of an image with different class if iou > 0.7"""
    txt_folder = Path(txt_folder)
    output_txt_folder = Path(output_txt_folder)
    output_txt_folder.mkdir(parents=True, exist_ok=True)

    def iou(boxA, boxB, image_size):
        # Convert normalized coordinates to pixel coordinates
        img_w, img_h = image_size
        boxA = [boxA[0] * img_w, boxA[1] * img_h, boxA[2] * img_w, boxA[3] * img_h]
        boxB = [boxB[0] * img_w, boxB[1] * img_h, boxB[2] * img_w, boxB[3] * img_h]
        # box = [x_min, y_min, width, height]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    # Add the merging logic here
    for txt_file in tqdm(txt_folder.glob("*.txt"), desc="Merging annotations"):
        anns = np.loadtxt(txt_file)
        # # if txt_file.stem != "glass_doors_00103":
        # if txt_file.stem != "office_doors_tester_11_0205123136_00003":
        # #     # print(anns)
        #     continue
        if len(anns.shape) == 1:
            if anns.shape[0] == 0:
                continue
            anns[0] = 0  # single annotation case
            # convert to yolo format - x-center, y-center, width, height from x-min, y-min, width, height
            anns[1] = anns[1] + anns[3] / 2
            anns[2] = anns[2] + anns[4] / 2
            out_file = output_txt_folder / txt_file.name
            np.savetxt(out_file, anns.reshape(1, -1), fmt="%d %.6f %.6f %.6f %.6f")
            continue

        anns = np.unique(anns, axis=0)  # remove duplicate rows
        imgp = input_folder / f"{txt_file.stem}.jpg"
        image = Image.open(imgp)
        final_anns = []
        skip_rows = set()
        merge_map = defaultdict(list)
        for i in range(len(anns) - 1):
            if i in skip_rows:
                continue
            for j in range(i + 1, len(anns)):
                if j in skip_rows:
                    continue
                if iou(anns[i][1:5], anns[j][1:5], image_size=image.size) > 0.7:
                    skip_rows.add(j)
                    merge_map[i].append(j)

        # sanity checks
        removed_rows = set()
        [removed_rows.update(f) for f in merge_map.values()]
        if sorted(removed_rows) != sorted(skip_rows):
            raise ValueError("Mismatch in removed rows and skip rows")

        final_anns = []
        for i, j_list in merge_map.items():
            merged_box = anns[i].copy()
            for j in j_list:
                # Simple average for merging boxes
                merged_box[1:5] = (merged_box[1:5] + anns[j][1:5]) / 2
            final_anns.append(merged_box)
        merged_rows = set(merge_map.keys())
        remaining_rows = set(range(len(anns))) - skip_rows - merged_rows
        for i in remaining_rows:
            final_anns.append(anns[i])
        out_file = output_txt_folder / txt_file.name
        final_anns = np.unique(np.array(final_anns), axis=0)  # remove duplicate rows
        final_anns[:, 0] = 0  # temporary fix for single annotation case
        # convert to yolo format - x-center, y-center, width, height from x-min, y-min, width, height
        final_anns[:, 1] = final_anns[:, 1] + final_anns[:, 3] / 2
        final_anns[:, 2] = final_anns[:, 2] + final_anns[:, 4] / 2
        np.savetxt(out_file, final_anns, fmt="%d %.6f %.6f %.6f %.6f")


def query_result_analysis(json_path):
    """Analyze query results from a JSON file."""
    with open(json_path, "r") as f:
        results = json.load(f)

    total_images = len(results)
    result_querys = {}
    for img, query_result in results.items():
        for query, has_object in query_result.items():
            if query not in result_querys:
                result_querys[query] = {"total": 0, "positive": 0, "negative": 0}
            result_querys[query]["total"] += 1
            if has_object:
                result_querys[query]["positive"] += 1
            else:
                result_querys[query]["negative"] += 1

    print(f"Total images: {total_images}")
    for query, stats in result_querys.items():
        print(f"Query: {query}")
        print(
            f"  Total: {stats['total']}, Positive: {stats['positive']}, Negative: {stats['negative']}"
        )
        print(f"  Positive Rate: {stats['positive'] / stats['total'] * 100:.2f}%")


def detection_results_analysis(detections_folder):
    """Analyze detection results from a folder of text files."""
    detections_folder = Path(detections_folder)
    total_images = 0
    total_detections = 0
    for txt_file in detections_folder.glob("*.txt"):
        anns = np.loadtxt(txt_file)
        if len(anns.shape) == 1:
            if anns.shape[0] == 0:
                continue
            total_detections += 1
            total_images += 1
            continue
        total_detections += anns.shape[0]
        total_images += 1

    print(f"images with detections: {total_images}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections / total_images:.2f}")


def filter_detections_by_query_result(
    detections_folder, query_results_json, output_folder
):
    """Filter detections based on query results and save to output folder."""
    detections_folder = Path(detections_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    with open(query_results_json, "r") as f:
        query_results = json.load(f)

    for txt_file in tqdm(detections_folder.glob("*.txt"), desc="Filtering detections"):
        image_name = txt_file.stem
        if image_name not in query_results or not any(
            query_results[image_name].values()
        ):
            continue  # skip images without the object

        anns = np.loadtxt(txt_file)
        out_file = output_folder / txt_file.name
        if len(anns.shape) == 1:
            if anns.shape[0] == 0:
                continue
            np.savetxt(out_file, anns.reshape(1, -1), fmt="%d %.6f %.6f %.6f %.6f")
            continue
        np.savetxt(out_file, anns, fmt="%d %.6f %.6f %.6f %.6f")
    detection_results_analysis(output_folder)


def moondream_viz(images_folder, annotations_folder, output_folder):
    """Visualize MoonDream detections by overlaying bounding boxes on images."""
    images_folder = Path(images_folder)
    annotations_folder = Path(annotations_folder)
    if not images_folder.exists() or not annotations_folder.exists():
        raise ValueError("Images folder or annotations folder does not exist.")
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for ann_file in tqdm(
        annotations_folder.glob("*.txt"), desc="Visualizing MoonDream Detections"
    ):
        image_file = images_folder / f"{ann_file.stem}.jpg"
        if not image_file.exists():
            print(f"Image file {image_file} does not exist. Skipping.")
            continue

        image = Image.open(image_file)
        draw = ImageDraw.Draw(image)

        anns = np.loadtxt(ann_file)
        if len(anns.shape) == 1:
            anns = anns[np.newaxis, :]  # single annotation case

        for ann in anns:
            # ann = [class_id, x_min, y_min, width, height] in normalized coordinates
            img_w, img_h = image.size
            x_min = ann[1] * img_w
            y_min = ann[2] * img_h
            x_max = (ann[1] + ann[3]) * img_w
            y_max = (ann[2] + ann[4]) * img_h
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

        output_image_path = output_folder / image_file.name
        image.save(output_image_path)
        # print(f"Saved visualization to {output_image_path}")


if __name__ == "__main__":
    # input_folder = Path('/data/datasets/doors_4492/train')
    # # output_folder = Path('/data/pavan/tycoai/rf-detr-jci/moondream_outputs/doors_4492/train/advanced')
    # output_folder = Path('/data/pavan/tycoai/rf-detr-jci/moondream_outputs/doors_4492/train/advanced_temp')

    # input_folder = Path('/data/datasets/p365_44k')
    # output_folder = Path('/data/pavan/tycoai/rf-detr-jci/moondream_outputs/p365_44k/query_only')

    # input_folder = Path('/data/datasets/intel_datasets/doors_jci/merged_selected_annotations6_common/COCO/val2017')
    # # output_folder = Path('/data/pavan/tycoai/rf-detr-jci/moondream_outputs/doors_jci/query_only_glass_doors')
    # output_folder = Path('/data/pavan/tycoai/rf-detr-jci/moondream_outputs/doors_jci/advanced')

    # input_folder = Path('/data/datasets/coco_val/images/')
    # output_folder = Path('/data/pavan/tycoai/rf-detr-jci/moondream_outputs/coco_val/advanced')

    # input_folder = Path('/media/fast_data_2/sreekanth/doors_datasets/doors_all_dataset/new_collection08122025/imnet1k_sliding_door/extracted_images')
    # output_folder = Path('/media/fast_data_1/pavan/codes/rf-detr-jci/moondream_outputs/imnet1k_sliding_door/advanced_v2')

    # input_folder = Path('/media/fast_data_2/sreekanth/doors_datasets/doors_all_dataset/new_collection08122025/craftsman_office/extracted_images')
    # output_folder = Path('/media/fast_data_1/pavan/codes/rf-detr-jci/moondream_outputs/craftsman_office/detections_only_v3')

    # input_folder = Path('/media/fast_data_2/sreekanth/doors_datasets/doors_all_dataset/new_collection08122025/imnet1k_sliding_door/extracted_images')
    # output_folder = Path('/media/fast_data_1/pavan/codes/rf-detr-jci/moondream_outputs/imnet1k_sliding_door/advanced_v2')

    # input_folder = Path('/media/fast_data_2/sreekanth/doors_datasets/doors_all_dataset/p365_44k')
    # output_folder = Path('/media/fast_data_1/pavan/codes/rf-detr-jci/moondream_outputs/p365_44k/advanced_v2')

    # input_folder = Path('/media/fast_data_2/sreekanth/doors_datasets/doors_all_dataset/new_collection08122025/honey_data_15m/extracted_images')
    # output_folder = Path('/media/fast_data_1/pavan/codes/rf-detr-jci/moondream_outputs/honey_data_15m/advanced_v1')

    input_folder = Path(
        "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/frames_filtered_v2_dedup"
    )
    output_folder = Path(
        "/data/pavan/tycoai/project_helpers/data_miner/output/projects/delivery_pov_v1/moondream/frames_filtered_v2_dedup"
    )

    ##################################################################################################
    ####################### Inference ################################################################

    # moondream = MoonDreamHelper(detection_class="door")
    # Process single image (original functionality)
    # process_image('/data/datasets/doors_4492/valid/objects365_v1_00333615.jpg', output_folder)

    moondream = MoonDreamHelper(
        query_only=False,
        reasoning=False,
        query=[
            "Is there a glass door in the image? Ignore any other doors like vehicle, cupboard or windows. The glass door has to be used for human entrance, does this image contain such a door? Answer yes or no."
        ],
        # query = "Is there a door in the image? Ignore vehicle doors. Answer yes or no."
        # detection_class=["door", "glass door"]
        detection_class=[
            "door or glass door, but dont detect window doors, cupboard doors, vehicle doors, or any other type of door except doors used for people entrance"
        ],
    )
    # Process entire folder
    moondream.process_folder(input_folder, output_folder)

    ##################################################################################################
    ####################### Postprocess ##############################################################

    merge_annotations(
        txt_folder=output_folder / "pred_txt",
        output_txt_folder=output_folder / "pred_txt_merged",
    )

    query_result_analysis(json_path=output_folder / "query_results_advanced.json")

    detection_results_analysis(detections_folder=output_folder / "pred_txt_merged")

    filter_detections_by_query_result(
        detections_folder=output_folder / "pred_txt_merged",
        query_results_json=output_folder / "query_results_advanced.json",
        output_folder=output_folder / "pred_txt_merged_filtered",
    )

    # difference_in_query_results(
    #     output_folder / 'query_results_with_reasoning.json',
    #     output_folder / 'query_results_no_reasoning.json'
    # )
    # print_false_negatives(
    #     output_folder / 'query_results_no_reasoning.json'
    # )
    ##########################################################################################
    ####################### VIZ ##############################################################

    # input_folder = Path('/data/datasets/doors_4492/train')
    # annotations_folder = Path('/data/pavan/tycoai/rf-detr-jci/moondream_outputs/doors_4492/train/advanced/pred_txt_merged')
    # output_folder = Path('/data/pavan/tycoai/rf-detr-jci/moondream_outputs/doors_4492/train/advanced/pred_txt_merged_viz_coco')

    # moondream_viz(
    #     images_folder=input_folder,
    #     annotations_folder=annotations_folder,
    #     output_folder=output_folder
    # )

    ##########################################################################################
    ####################### TEMP ##############################################################
    # detection_results_analysis(
    #     detections_folder='/media/fast_data_1/pavan/codes/rf-detr-jci/moondream_outputs/craftsman_office/advanced/pred_txt_merged/'
    # )
    # detection_results_analysis(
    #     detections_folder='/media/fast_data_1/pavan/codes/rf-detr-jci/moondream_outputs/craftsman_office/detections_only/pred_txt_merged/'
    # )
    # detection_results_analysis(
    #     detections_folder='/media/fast_data_1/pavan/codes/rf-detr-jci/moondream_outputs/craftsman_office/detections_only_v2/pred_txt_merged/'
    # )
