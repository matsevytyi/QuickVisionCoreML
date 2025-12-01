import Foundation
import CoreML
import CoreImage
import CoreVideo
import CoreGraphics

public struct Detection {
    public let bbox: CGRect   // in normalized [0,1] coordinates relative to model input
    public let confidence: Float
    public let classIndex: Int
    public let className: String?

    public init(
        bbox: CGRect,
        confidence: Float,
        classIndex: Int,
        className: String? = nil
    ) {
        self.bbox = bbox
        self.confidence = confidence
        self.classIndex = classIndex
        self.className = className
    }
}

public final class QuickObjectDetectionModel {
    
    // MARK: Properties
    
    private let model: MLModel
    
    // Input
    private var inputName: String = "image" // for YOLO input is 'image', Image (Color 640 × 640)
    private var inputWidth: Int = 640 // for DETR input is 'image_input', MultiArray (Float16 1 × 3 × 512 × 512)
    private var inputHeight: Int = 640
    
    private var IoUThreshold: Double = 0.45
    private var confidenceThreshold: Double = 0.25 
    
    // Output
    
    // for YOLO
// named 'confidence', MultiArray (Float32 0 × 80), [0...x80] Boxes x Class confidence (see user-defined metadata "classes")
// named 'coordinates', MultiArray (Float32 0 × 4), [0...x4] Boxes × [x, y, width, height] (relative to image size)
    
    // for DETR
// named 'boxes', MultiArray (Float16 1 × 300 × 4)
// named 'scores', MultiArray (Float16 1 × 300)
// named 'label', MultiArray (Float16 1 × 300)
    
    private var outputConfName: String = "coordinates"
    private var outputLocationName: String = "output"
    private var outputLabelName: String? // not for YOLO - only if class
    
    private var outputConfShape: [Int]
    private var outputLocationShape: [Int]
    private var outputLabelShape: [Int]
    
    private var classLabels: [Any]?
    
    // MARK: Init
    
    /// Automatic metadata-based configuration
    public init(model: MLModel) throws {
        self.model = model
        
        let desc = model.modelDescription
        
        let inputs = desc.inputDescriptionsByName
        self.inputName = inputs.keys.first { $0.contains("image") || $0 == "input" || inputs[$0]?.multiArrayConstraint != nil } ?? "image"
        self.inputWidth = inputs[self.inputName]?.imageConstraint?.pixelsWide ?? 640
        self.inputHeight = inputs[self.inputName]?.imageConstraint?.pixelsHigh ?? 640
        print("Extracted input: \(self.inputName), \(self.inputWidth)/\(self.inputWidth)")
        
        let outputs = desc.outputDescriptionsByName
        
        // Debug
        // print("Outputs: \(outputs.map { "\($0.key): \($0.value) / \($0.value.name) / \($0.value.multiArrayConstraint?.shape)" })") // this way $0.value.multiArrayConstraint?.shape extracts expected shape of MultiArray, but where do I use it? i.e. Optional([0, 80])
        
        self.outputLocationName = outputs.keys.first { $0.contains("coord") || $0.contains("box") } ?? "coordinates"
        self.outputConfName = outputs.keys.first { $0.contains("scor") || $0.contains("conf") } ?? "confidence"
        self.outputLabelName = outputs.keys.first { $0.contains("label") || $0.contains("clas") } ?? nil //not everywhere by default
        
        self.outputConfShape = outputs[self.outputConfName]?.multiArrayConstraint?.shape as? [Int] ?? []
        self.outputLocationShape = outputs[self.outputLocationName]?.multiArrayConstraint?.shape as? [Int] ?? []
        
        if let outputLabelName = self.outputLabelName,
           let outputLabel = outputs[outputLabelName] {
            
            self.outputLabelShape = outputLabel.multiArrayConstraint?.shape as? [Int] ?? []
        } else {
            self.outputLabelShape = []
        }
        
        print("Extracted output conf: \(self.outputConfName), \(self.outputConfShape))")
        print("Extracted output loc: \(self.outputLocationName), \(self.outputLocationShape)")
        print("Extracted output labels: \(self.outputLabelName), \(self.outputLabelShape)")
        
        self.classLabels = desc.classLabels
    }
    
    /// Manual configuration override with automatic model detection on fallback
    public convenience init(model: MLModel, config: [String: Any]) throws {
        try self.init(model: model)
        if let inputName = config["inputName"] as? String { self.inputName = inputName }
        if let inputWidth = config["inputWidth"] as? Int { self.inputWidth = inputWidth }
        if let inputHeight = config["inputHeight"] as? Int { self.inputHeight = inputHeight }
        
        if let outputConfName = config["outputConfName"] as? String { self.outputConfName = outputConfName }
        if let outputLocationName = config["outputLocationName"] as? String { self.outputLocationName = outputLocationName }
        if let outputLabelName = config["outputLabelName"] as? String? { self.outputLabelName = outputLabelName }
        
        if let IoUThreshold = config["IoUThreshold"] as? Double { self.IoUThreshold = IoUThreshold }
        if let confidenceThreshold = config["confidenceThreshold"] as? Double { self.confidenceThreshold = confidenceThreshold }

        if let outputConfShape = config["outputConfShape"] as? [Int] { self.outputConfShape = outputConfShape }
        if let outputLocationShape = config["outputLocationShape"] as? [Int] { self.outputLocationShape = outputLocationShape }
        if let outputLabelShape = config["outputLabelShape"] as? [Int] { self.outputLabelShape = outputLabelShape }

        if let classLabels = config["classLabels"] as? [String] { self.classLabels = classLabels }
    }
    
    // MARK: Public API (2 functions)
    
    public func predict(image: CGImage) -> [Detection] {
        do {
            let pixelBuffer = try preprocessImage(image)
            return try predictHelper(pixelBuffer: pixelBuffer)
        } catch {
            print("[QuickObjectDetectionModel] predict(image:) error: \(error)")
            return []
        }
    }
    
    public func predict(pixelBuffer: CVPixelBuffer) -> [Detection] {
        do {
            let resized = try resizePixelBuffer(pixelBuffer)
            return try predictHelper(pixelBuffer: resized)
        } catch {
            print("[QuickObjectDetectionModel] predict(pixelBuffer:) error: \(error)")
            return []
        }
    }
    
    // MARK: Core implementation
    
    private func predictHelper(pixelBuffer: CVPixelBuffer) throws -> [Detection] {
        let inputValue = MLFeatureValue(pixelBuffer: pixelBuffer)
        let input = try MLDictionaryFeatureProvider(dictionary: [inputName: inputValue])
        
        let prediction = try model.prediction(from: input)
        
        guard let locationArray = prediction.featureValue(for: outputLocationName)?.multiArrayValue,
              let confArray = prediction.featureValue(for: outputConfName)?.multiArrayValue else {
            print("[QuickObjectDetectionModel] No MultiArray output for key \(outputLocationName) or \(outputConfName)")
            return []
        }
        
        // Dispatch based on conf shape (YOLO: 2D [N,classes], DETR: 1D [1,N])
        return outputConfShape.count == 2 ? parseYOLO(locationArray, confArray) : parseDETR(locationArray, confArray, prediction)
    }

    // resize helpers
    private func resizePixelBuffer(_ buffer: CVPixelBuffer) throws -> CVPixelBuffer {
        var outputBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            inputWidth,
            inputHeight,
            CVPixelBufferGetPixelFormatType(buffer),
            attrs,
            &outputBuffer
        )
        
        guard status == kCVReturnSuccess, let resizedBuffer = outputBuffer else {
            throw NSError(domain: "PixelBufferResize",
                          code: 0,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to create pixel buffer"])
        }
        
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        CVPixelBufferLockBaseAddress(resizedBuffer, [])
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(resizedBuffer, [])
        }
        
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let ciContext = CIContext()
        
        let sx = CGFloat(inputWidth) / CGFloat(CVPixelBufferGetWidth(buffer))
        let sy = CGFloat(inputHeight) / CGFloat(CVPixelBufferGetHeight(buffer))
        let scaleTransform = CGAffineTransform(scaleX: sx, y: sy)
        let resizedCIImage = ciImage.transformed(by: scaleTransform)
        
        ciContext.render(resizedCIImage, to: resizedBuffer)
        
        return resizedBuffer
    }
    
    private func preprocessImage(_ cgImage: CGImage) throws -> CVPixelBuffer {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            inputWidth,
            inputHeight,
            kCVPixelFormatType_32ARGB,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw NSError(domain: "ImageProcessing",
                          code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to create pixel buffer"])
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: inputWidth,
            height: inputHeight,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            throw NSError(domain: "ImageProcessing",
                          code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to create CGContext"])
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: inputWidth, height: inputHeight))
        
        return buffer
    }
    
    // parse helpers
    private func parseYOLO(_ boxes: MLMultiArray, _ scores: MLMultiArray) -> [Detection] {
        guard boxes.shape.count == 2, scores.shape.count == 2,
              Int(boxes.shape[0].doubleValue) == Int(scores.shape[0].doubleValue) else {
            print("[Parse YOLO] <QuickObjectDetectionModel> YOLO shape mismatch: boxes \(boxes.shape), scores \(scores.shape)")
            return []
        }
        
        let numBoxes = Int(boxes.shape[0].doubleValue)
        let numClasses = Int(scores.shape[1].doubleValue)
        var detections: [Detection] = []
        
        for i in 0..<numBoxes {
            var maxScore: Float = 0
            var bestClass = 0
            
            for c in 0..<numClasses {
                let score = (try? scores[[NSNumber(value: i), NSNumber(value: c)]].floatValue) ?? 0
                if score > maxScore {
                    maxScore = score
                    bestClass = c
                }
            }
            
            guard maxScore > Float(confidenceThreshold) else { continue }
            
            let cx = (try? boxes[[NSNumber(value: i), NSNumber(value: 0)]].doubleValue) ?? 0
            let cy = (try? boxes[[NSNumber(value: i), NSNumber(value: 1)]].doubleValue) ?? 0
            let w = (try? boxes[[NSNumber(value: i), NSNumber(value: 2)]].doubleValue) ?? 0
            let h = (try? boxes[[NSNumber(value: i), NSNumber(value: 3)]].doubleValue) ?? 0
            
            let x1 = cx - w / 2
            let y1 = cy - h / 2
            
            detections.append(Detection(
                bbox: CGRect(x: x1, y: y1, width: w, height: h),
                confidence: maxScore,
                classIndex: bestClass,
                className: classLabels?[bestClass] as? String
            ))
        }
        return detections
    }

    private func parseDETR(_ boxes: MLMultiArray, _ scores: MLMultiArray, _ prediction: MLFeatureProvider) -> [Detection] {
        guard boxes.shape.count == 3, scores.shape.count == 2,
              Int(boxes.shape[1].doubleValue) == 300 else {
            print("[Parse DETR] <QuickObjectDetectionModel> DETR shape mismatch: boxes \(boxes.shape), scores \(scores.shape)")
            return []
        }
        
        var detections: [Detection] = []
        let numDets = 300
        
        let labelArray: MLMultiArray?
        if let labelName = outputLabelName,
           let labels = prediction.featureValue(for: labelName)?.multiArrayValue {
            labelArray = labels
        } else {
            labelArray = nil
        }
        
        for i in 0..<numDets {
            let score = (try? scores[[NSNumber(value: 0), NSNumber(value: i)]].floatValue) ?? 0
            guard score > Float(confidenceThreshold) else { continue }
            
            let x1 = (try? boxes[[NSNumber(value: 0), NSNumber(value: i), NSNumber(value: 0)]].doubleValue) ?? 0
            let y1 = (try? boxes[[NSNumber(value: 0), NSNumber(value: i), NSNumber(value: 1)]].doubleValue) ?? 0
            let x2 = (try? boxes[[NSNumber(value: 0), NSNumber(value: i), NSNumber(value: 2)]].doubleValue) ?? 1
            let y2 = (try? boxes[[NSNumber(value: 0), NSNumber(value: i), NSNumber(value: 3)]].doubleValue) ?? 1
            
            let classIdx: Int
            if let labelArray = labelArray, labelArray.shape.count == 2 {
                classIdx = Int((try? labelArray[[NSNumber(value: 0), NSNumber(value: i)]].doubleValue) ?? 0)
            } else {
                classIdx = 0
            }
            
            detections.append(Detection(
                bbox: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1),
                confidence: score,
                classIndex: classIdx,
                className: classLabels?[classIdx] as? String
            ))
        }
        return detections
    }

}

