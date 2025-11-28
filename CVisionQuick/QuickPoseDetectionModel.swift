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
    
    private let inputName: String
    private let inputWidth: Int
    private let inputHeight: Int
    
    private let outputName: String
    
    
    public init(model: MLModel) throws {
        
        self.model = model
        
        let desc = self.model.modelDescription
        
        // Determine input feature name and size
        if let (name, feature) = desc.inputDescriptionsByName.first,
               feature.type == .image,
               let constraint = feature.imageConstraint {
                self.inputName = name
                self.inputWidth = constraint.pixelsWide
                self.inputHeight = constraint.pixelsHigh
            } else {
                // Fallbacks (default appoach - більшість користується 640/640)
                self.inputName = desc.inputDescriptionsByName.keys.first ?? "image"
                self.inputWidth = 640
                self.inputHeight = 640
            }
        
        // Output feature type
        if let (name, _) = desc.outputDescriptionsByName.first {
                self.outputName = name
            } else {
                throw NSError(
                    domain: "QuickPoseDetectionModel",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to extract output feature metadata, make sure the model is .mlmodel. If problem persists, specify settings manually."]
                )

            }
        
        // TODO: POSSIBLE ADDITIONS
        // detect image/video encoding
        
        // Number of keypoints (currently 17).
        // Number of channels (currently 56) and semantics (4+1+51).
        // Confidence thresholds (0.5).
        // Whether output coordinates are normalized or absolute.
        // Expected output shape (e.g. [1, 56, 8400])
        
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
            
            print("Buffer received")
            let resizedBuffer = try resizePixelBuffer(pixelBuffer, width: 640, height: 640)
            
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
        return parseCocoOutput(prediction)
    }
    
    private func resizePixelBuffer(_ buffer: CVPixelBuffer, width: Int, height: Int) throws -> CVPixelBuffer {
        var outputBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
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
        let scaleTransform = CGAffineTransform(scaleX: CGFloat(width) / CGFloat(CVPixelBufferGetWidth(buffer)),
                                               y: CGFloat(height) / CGFloat(CVPixelBufferGetHeight(buffer)))
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
        guard let multiArray = prediction.featureValue(for: self.outputName)?.multiArrayValue else {
            print("No multiArray output")
            return []
        }
        
        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(multiArray.dataPointer))
        
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
            let x = ptr[(5 + kp * 3) * anchors + bestIndex]
            let y = ptr[(5 + kp * 3 + 1) * anchors + bestIndex]
            let conf = ptr[(5 + kp * 3 + 2) * anchors + bestIndex]
            
            if conf > 0.5 {
                keypoints.append(CGPoint(x: CGFloat(x), y: CGFloat(y)))
                print("normal kp _\(kp) with \(x), \(y)")
            } else {
                keypoints.append(CGPoint(x: 0, y: 0))
                print("abnormal kp _ \(kp)")
            }
        }
        
        return keypoints
    }
}
