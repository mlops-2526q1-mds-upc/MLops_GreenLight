---
pretty_name: "Bosch Small Traffic Lights Dataset"
language:
  - en
tags:
  - image
  - object_detection
  - traffic_lights
  - small_objects
license: “non-commercial use only”  # as per original
task_categories:
  - object_detection
size_categories:
  - small
  - medium
features:
  - name: img
    dtype: image
  - name: boxes
    dtype:
      sequence:
        - name: label
          dtype: string
        - name: occluded
          dtype: bool
        - name: x_min
          dtype: float32
        - name: y_min
          dtype: float32
        - name: x_max
          dtype: float32
        - name: y_max
          dtype: float32
splits:
  - name: train
    num_examples: 5093
  - name: test
    num_examples: 8334
download_size: [More Information Needed]
dataset_size: [More Information Needed]
---

# Dataset Description

The Bosch Small Traffic Lights Dataset (BSTLD) is an image dataset for object detection and classification of traffic lights in urban driving scenes. It contains high-resolution images (1280×720) with bounding boxes for traffic lights and their active state (e.g., red, yellow, green, off). The dataset is challenging due to small object sizes (median width ~8 to 9 pixels), changing illumination, occlusions, and visual clutter.

---

## Dataset Structure

There are more than 13,000 images total, of which the dataset includes both annotated frames and non-annotated (empty) images. The annotations include traffic light *state*, occlusion flag, and bounding box coordinates. The train and test splits are geographically and temporally independent to some extent.

---

## Dataset Creation

- **Source**: Collected by Bosch, using cameras in various driving conditions.
- **Motivation**: To provide a dataset that challenges detection of small objects (traffic lights), with realistic difficulties: lighting, occlusion, changing exposure, multiple visible lights, etc.
- **Annotation process**: Human annotators labeled bounding boxes of traffic lights, marking the *visible (active) state*, and whether each light is *occluded*. Some test frames were annotated frame-by-frame (every frame) to capture temporal consistency.

---

## Considerations for Using the Data

- Many traffic lights are very small (≈8 pixels width), which poses a challenge for detection models; high resolution and fine features matter.  
- Color conversion from raw HDR images to 8-bit RGB may introduce artifacts or unusual color distributions; this might affect color-based features.
- Occlusion: some traffic lights are partially occluded; ground truth for occlusion might vary in quality.  
- Not all classes are equally represented. For example, the “off” class or certain directional states may have far fewer examples.  
- The dataset was not designed for production deployment; its geographic / lighting conditions may not cover all real-world variation.  

---

## Usage

You can load this dataset using the `datasets` library via a YAML loader or custom Dataset class. Use the splits (train / test) provided. For object detection tasks, bounding boxes + state labels + occlusion flag are the main targets. Because multiple labels exist for directional/state variants, it is common to *alias* them to fewer classes (e.g. Red, Yellow, Green, Off) to reduce class imbalance and simplify state classification.

---

## Additional Information

- **Citations**


