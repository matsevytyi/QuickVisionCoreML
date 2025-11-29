# ``QuickVisionCoreML``

A Swift framework for **rapid CoreML Computer Vision model deployment** on iOS 16+. Load any  model and easily predict keypoints from `CGImage`/`CVPixelBuffer`, framework handles resizing and output parsing automatically based on extracted .mlmodel netadata.


## Overview

``QuickPoseDetectionModel`` allows to quickly deploy and PoC new .mlmodels, allowing to abstract yourself from technical details like buffer/image resizing or output parsing.

- on `init(model: MLModel)` it auto-detects model metadata (I/O size and shape) and heuristically determines output type (YOLO-like/Heatmap - `8400 anchors 56 channels` or `[K, H, W]`). If metadata is missing it will default to YOLO-like settings as the most common

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

3. Run prediction 
`let keypoints = poseDetector.predict(image: cgImage) // Returns [CGPoint] (normalized 0-1)`

4. When visualising, multiply by screen width
```
Circle()
    .fill(Color.green)
    .frame(width: 8, height: 8)
    .position(
        x: (1 - point.x) * geometry.size.width,
        y: point.y * geometry.size.height
    )
```

## Topics

### <!--@START_MENU_TOKEN@-->Group<!--@END_MENU_TOKEN@-->

- <!--@START_MENU_TOKEN@-->``Symbol``<!--@END_MENU_TOKEN@-->
