# QuickVisionCoreML

A Swift framework for **rapid CoreML Computer Vision model deployment** on iOS 16+. Load any  model and easily predict keypoints from `CGImage`/`CVPixelBuffer`, framework handles resizing and output parsing automatically based on extracted .mlmodel netadata.


## Overview

Framework consists of few classes, each allowing to quickly deploy and PoC new .mlmodels, therefore, abstracting a user from technical details like buffer/image resizing or output parsing.

- on `init(model: MLModel)` it auto-detects model metadata (I/O size, shape, variables) and heuristically determines output type/shape to properly convert data in user-firendly still reliable way. If metadata is missing it will default to YOLO-like settings as the most common

- if a user wants to customize model, they may use `init(model: MLModel, config: [String: Any])`. This way, the model metadata is extracted and then we attempt to overwrite user-specified settings.

- `predict()` allows to full prediction logic (including all necessary tech details) in one line. For convenience there is (a) `predict(pixelBuffer: CVPixelBuffer)` if you use a stream from camera, and (b) `predict(image: CGImage)` if you want to test on a picture. In both cases it returns coordinates in [0...1] range, 

## Quick Start

1. Convert your model to .mlmodel, for example with `coremltools`

2. Connect your model:

```
let config = MLModelConfiguration()
let rawCoreMLModel = try yolov8n_pose_model(configuration: config)
self.model = try QuickPoseDetectionModel(model: rawCoreMLModel.model)
```

OR

`self.model = try QuickPoseDetectionModel(model: yolov8n_pose_model(configuration: MLModelConfiguration()).model)`

3. Run prediction 
`let keypoints = poseDetector.predict(image: cgImage) // Returns [CGPoint] (normalized 0-1)`

4. When visualising, multiply coordinates by screen dimensions (i.e. `x = point.x * geometry.size.width`)

## Full available functionality

### Object Pose Prediction

Available on ``QuickPoseDetectionModel``, supports 2 main classes (YOLO/regression-like and heatmap-like)

- on `init(model: MLModel)` it auto-detects model metadata (I/O size and shape) and heuristically determines output type (YOLO-like/Heatmap - `8400 anchors 56 channels` or `[K, H, W]`). If metadata is missing it will default to YOLO-like settings as the most common

-  There is available `init(model: MLModel, config: [String: Any])`, `predict(pixelBuffer: CVPixelBuffer)` and `predict(image: CGImage)`; both work same way as listed in OverView

### Object Detection

Available on ``QuickObjectDetectionModel``, supports Transformer(DETR) and YOLO-like inputs and outputs

- on `init(model: MLModel)` it auto-detects model metadata (I/O size, shape, variables) and heuristically determines output type (YOLO `coordinates`/`confidence` or DETR `boxes`/`scores`). If metadata is missing it will default to YOLO-like settings as the most common

- There is available `init(model: MLModel, config: [String: Any])`, `predict(pixelBuffer: CVPixelBuffer)` and `predict(image: CGImage)`; both work same way as listed in Overview

### Object Classification

Available on ``QuickObjectClassificationModel``, supports nearly all image classification models as they mostly function similar way

- on `init(model: MLModel)` it auto-detects model metadata (I/O size and shape) and heuristically determines output type (string label or `classLabelProbs`/logits). If metadata is missing it will default to common classification settings

- There is available `init(model: MLModel, config: [String: Any])`, `predict(pixelBuffer: CVPixelBuffer)` and `predict(image: CGImage)`; both work same way as listed in Overview

## Current Limitations & Future Work

`0.2.1` – Currently framework doesnt support INT8 and FP16 quantizations, it future it should be determined and done on the fly

`0.2.2` – There should be added option to automaticaly obtain model output in user-defined dataclass or automatically generated/user-defined SwiftUI/UIKit View

`0.2.3` – Now there are no device-specific accelerations, they should be added as they may influence model choice at PoC testing

`0.3.0` – Extend support towards Object Segmentation and Depth Estimation. Depending on the model there are significant complications (i.e. if segmentation model is semantic, instance or panoptic the way output should be handled differs significantly)

`0.4.0` – Extend support towards other types of Vision2Vision, Text2Vision, Vision2Text and other multimodal tasks. – full list available at https://huggingface.co/models ? computer vision

Also, there should be considered such features:
- Upload and test models from huggingface/kaggle with a single line
- Have some popular models already pre-uploaded
- One-line training, evaluation and finetuning support for models
