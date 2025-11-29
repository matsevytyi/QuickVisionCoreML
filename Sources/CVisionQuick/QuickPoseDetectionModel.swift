//
//  QuickPoseDetectionModel.swift
//  CVisionQuick
//
//  Created by Andrii Matsevytyi on 28.11.2025.
//
import Foundation
import CoreML
import Accelerate
import CoreImage

public class QuickPoseDetectionModel {
    
    private let model: MLModel
    
    // input features
    private let inputName: String
    private let inputWidth: Int
    private let inputHeight: Int
    
    // output features
    private var outputName: String
    private var isHeatmapModel: Bool = false
    private var numKeypoints: Int = 17
    private var heatmapSize: (Int, Int) = (0, 0)
    private var outputStride: Float = 1.0
    
    // other features
    private var detectionThreshold: Float = 0.5
    
    public init(model: MLModel) throws {
        
        self.model = model
        
        let desc = self.model.modelDescription
        
        // Determine input feature name and size
        if let (inputName, inputFeature) = desc.inputDescriptionsByName.first,
           inputFeature.type == .image,
           let constraint = inputFeature.imageConstraint {
            self.inputName = inputName
            self.inputWidth = constraint.pixelsWide
            self.inputHeight = constraint.pixelsHigh
        } else {
            // Fallbacks (default appoach - більшість користується 640/640)
            self.inputName = desc.inputDescriptionsByName.keys.first ?? "image"
            self.inputWidth = 640
            self.inputHeight = 640
            print("Failed to extract output feature metadata, make sure the model is .mlmodel. If problem persists, specify settings manually.")
            
        }
        
        // Output feature type and sizes
        guard let (outputName, outputFeature) = desc.outputDescriptionsByName.first,
              outputFeature.type == .multiArray
        else {
            throw NSError(domain: "QuickPoseDetectionModel",
                          code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No MultiArray output"])
        }
        
        self.outputName = outputName
        
        // shape extraction attempt
        let shape = try getOutputShape(desc: desc, outputName: outputName)
        self.isHeatmapModel = isHeatmapShape(shape)  // [K, H, W]
        self.numKeypoints = extractKeypoints(shape)
        self.heatmapSize = isHeatmapModel ? (shape[1], shape[2]) : (0, 0)
        self.outputStride = calculateStride(inputSize: (inputWidth, inputHeight),
                                            outputSize: isHeatmapModel ? (shape[1], shape[2]) : nil)
        
        // TODO: POSSIBLE ADDITIONS
        // detect image/video encoding

        // Confidence thresholds (0.5).!!!
        
        // detect specific hardware accelerations
    }
    
    public func predict(image: CGImage) -> [CGPoint] {
        
        do {
            
            let pixelBuffer = try preprocessImage(image)

            return try predictHelper(pixelBuffer: pixelBuffer)
            
        } catch {
            print("Error extracting keypoints: \(error)")
            return []
        }
        
    }
    
    public func predict(pixelBuffer: CVPixelBuffer) -> [CGPoint] {
        do {
            
            print("Buffer received", pixelBuffer)
            let resizedBuffer = try resizePixelBuffer(pixelBuffer)
            
            print("Buffer resized")
            
            return try predictHelper(pixelBuffer: resizedBuffer)
            
        } catch {
            print("Error extracting keypoints: \(error)")
            return []
        }
    }
    
    private func predictHelper(pixelBuffer: CVPixelBuffer) throws -> [CGPoint] {
        
        // Wrap CVPixelBuffer into MLFeatureProvider using the expected input name.
        
        let inputValue = MLFeatureValue(pixelBuffer: pixelBuffer)
        let input = try MLDictionaryFeatureProvider(dictionary: [self.inputName: inputValue])
        
        print("Raw input", input)

        // Core ML prediction
        let prediction = try model.prediction(from: input)
        
        print("Raw output", prediction)

        // Extract keypoints
        if self.isHeatmapModel {
            // Heatmap style
            return parseHeatmapOutput(prediction)
        } else {
            // Default (YOLO/Coco) style
            return parseCocoOutput(prediction)
        }
    }
    
    private func resizePixelBuffer(_ buffer: CVPixelBuffer) throws -> CVPixelBuffer {
        var outputBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            self.inputWidth,
            self.inputHeight,
            CVPixelBufferGetPixelFormatType(buffer),
            attrs,
            &outputBuffer
        )

        guard status == kCVReturnSuccess, let resizedBuffer = outputBuffer else {
            throw NSError(domain: "PixelBufferResize", code: 0, userInfo: [NSLocalizedDescriptionKey: "Failed to create pixel buffer"])
        }

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        CVPixelBufferLockBaseAddress(resizedBuffer, [])
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(resizedBuffer, [])
        }

        // Use Core Image for fast resizing
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let ciContext = CIContext()
        let scaleTransform = CGAffineTransform(scaleX: CGFloat(self.inputWidth) / CGFloat(CVPixelBufferGetWidth(buffer)),
                                               y: CGFloat(self.inputHeight) / CGFloat(CVPixelBufferGetHeight(buffer)))
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
            self.inputWidth,
            self.inputHeight,
            kCVPixelFormatType_32ARGB,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw NSError(domain: "ImageProcessing", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create pixel buffer"])
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: self.inputWidth,
            height: self.inputHeight,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            throw NSError(domain: "ImageProcessing", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGContext"])
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: self.inputWidth, height: self.inputHeight))
        
        return buffer
    }
    
    private func parseCocoOutput(_ prediction: MLFeatureProvider) -> [CGPoint] {
        guard let outputArray = prediction.featureValue(for: self.outputName)?.multiArrayValue else {
            print("No multiArray output")
            return []
        }
        
        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(outputArray.dataPointer))
        
        let channels = 56  // 4 bbox + 1 obj + 51 keypoints
        let anchors = 8400
        
        var bestObjectness: Float32 = 0
        var bestIndex: Int = -1
        
        // Find anchor with highest objectness (channel index 4)
        for i in 0..<anchors {
            let obj = ptr[4 * anchors + i]
            if obj > bestObjectness {
                bestObjectness = obj
                bestIndex = i
            }
        }
        
        guard bestIndex != -1, bestObjectness > 0.5 else {
            print("No object detected")
            return []
        }
        
        // Extract 17 keypoints (from channel 5 to 55)
        var keypoints: [CGPoint] = []
        
        for kp in 0..<17 {
            let x = ptr[(5 + kp * 3) * anchors + bestIndex] / Float(self.inputWidth)
            let y = ptr[(5 + kp * 3 + 1) * anchors + bestIndex] / Float(self.inputHeight)
            let conf = ptr[(5 + kp * 3 + 2) * anchors + bestIndex]
            
            if conf > self.detectionThreshold {
                keypoints.append(CGPoint(x: CGFloat(x), y: CGFloat(y)))
                print("normal kp _\(kp) with \(x), \(y)")
            } else {
                keypoints.append(CGPoint(x: 0, y: 0))
                print("abnormal kp _\(kp) with \(x), \(y) and conf=\(conf)")
            }
        }
        
        return keypoints
    }

    private func parseHeatmapOutput(_ prediction: MLFeatureProvider) -> [CGPoint] {
        
        guard let outputArray = prediction.featureValue(for: self.outputName)?.multiArrayValue else {
            print("No multiArray output")
            return []
        }
        
        let shape = outputArray.shape.map { Int(truncating: $0) } // [K, H, W]
        
        let keypoints: [CGPoint] = (0..<numKeypoints).map { k in
            var maxVal: Float = -Float.infinity
            var maxY = 0, maxX = 0
            
            // Find argmax in heatmap slice [H,W] for keypoint k
            for y in 0..<shape[1] {
                for x in 0..<shape[2] {
                    let val = outputArray[[NSNumber(value: k),
                                     NSNumber(value: y),
                                     NSNumber(value: x)]].floatValue
                    if val > maxVal {
                        maxVal = val
                        maxY = y
                        maxX = x
                    }
                }
            }
            
            let relativeX = CGFloat(Float(maxX) / Float(self.inputWidth) * self.outputStride)
            let relativeY = CGFloat(Float(maxY) / Float(self.inputHeight) * self.outputStride)

            if maxVal > self.detectionThreshold {
                print("normal kp _\(k) at (\(relativeX), \(relativeY))")
            } else {
                print("abnormal kp _\(k) at (\(relativeX), \(relativeY)) with conf=\(maxVal)")
            }
            
            return CGPoint(x: relativeX, y: relativeY)
            
        }
        
        return keypoints
    }


    
    // MARK: init helpers
    
    private func getOutputShape(desc: MLModelDescription, outputName: String) throws -> [Int] {
        
        guard let output = desc.outputDescriptionsByName[outputName] else {
            throw NSError(domain: "QuickPoseDetectionModel", code: 1001, userInfo: [NSLocalizedDescriptionKey : "Could not find output description for \(outputName)"])
        }
        
        guard let constraint = output.multiArrayConstraint else {
            return []
        }
        
        return constraint.shape.map {
            Int(truncating: $0)
        }
        
    }

    private func isHeatmapShape(_ shape: [Int]) -> Bool {
        // Heatmap: [K, H, W] may contain K=14/17 (keypoints first)
        return shape.count == 3 && (14...17).contains(shape[0])
    }

    private func extractKeypoints(_ shape: [Int]) -> Int {
        
        // attempt heatmap
        if shape.count == 3 && (14...17).contains(shape[0]) {
            return shape[0]
        }
        
        return 17 // YOLO default
    }

    // downsampling factor between last hidden layer (usually Convolutional) and output layer
    private func calculateStride(inputSize: (Int, Int), outputSize: (Int, Int)?) -> Float {
        
        guard let out = outputSize else { return 1.0 }
        
        let res = Float(inputSize.0 / out.0)
        
        return res
    }
    
}
