//
//  CVisionQuick.swift
//  CVisionQuick
//
//  Created by Andrii Matsevytyi on 28.11.2025.
//

import Foundation
import CoreML

import Accelerate
import CoreVideo
import CoreImage

public class API {
    public init() {}
    public func doSomething() -> String {
        return "I am new framework"
    }
}

public class QuickPoseDetectionModel {
    private let model: MLModel
    
    public init(model: MLModel) {
        self.model = model
        
        // extract input name and dimensions
        // exctract output name and dimensions
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
        // "image" and "1035" myst be replaced from hardcoding to extracting out from model metadata
        let inputValue = MLFeatureValue(pixelBuffer: pixelBuffer)
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": inputValue])
        
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
        let width = 640
        let height = 640
        
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
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
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            throw NSError(domain: "ImageProcessing", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGContext"])
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return buffer
    }
    
    private func parseCocoOutput(_ prediction: MLFeatureProvider) -> [CGPoint] {
        guard let multiArray = prediction.featureValue(for: "var_1035")?.multiArrayValue else {
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

    
//    private func parseCocoOutput(_ prediction: MLFeatureProvider) -> [CGPoint] {
//        // Get YOLOv8 pose output: 1 x 56 x 8400
//        guard let tensor = prediction.featureValue(for: "var_1035")?.multiArrayValue else {
//            print("[PARSE OUTPUT] feature not found for key: 'var_1035'")
//            return []
//        }
//
//        guard tensor.dataType == .float32,
//              tensor.shape.count == 3,
//              tensor.shape[1].intValue >= 56
//        else {
//            print("[PARSE OUTPUT] unexpected tensor shape: \(tensor.shape)")
//            return []
//        }
//
//        let channels = tensor.shape[1].intValue   // 56
//        let anchors  = tensor.shape[2].intValue   // 8400
//
//        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(tensor.dataPointer))
//
//        // 1) find best anchor by highest objectness at channel 4
//        var bestObjectness: Float32 = 0
//        var bestIndex: Int = -1
//        let objChannel = 4
//
//        for i in 0..<anchors {
//            let obj = ptr[objChannel * anchors + i]
//            if obj > bestObjectness {
//                bestObjectness = obj
//                bestIndex = i
//            }
//        }
//
//        guard bestIndex != -1, bestObjectness > 0.5 else {
//            print("[PARSE OUTPUT] no object with sufficient confidence")
//            return []
//        }
//
//        // 2) extract 17 keypoints from channels 5...(5+3*17-1)
//        var keypoints: [CGPoint] = []
//        keypoints.reserveCapacity(17)
//
//        for kp in 0..<17 {
//            let baseChannel = 5 + kp * 3
//
//            let x = ptr[baseChannel * anchors + bestIndex]
//            let y = ptr[(baseChannel + 1) * anchors + bestIndex]
//            let conf = ptr[(baseChannel + 2) * anchors + bestIndex]
//
//            if conf > 0.5 {
//                // Normalize to 0–1 relative to 640×640 input
//                keypoints.append(CGPoint(x: CGFloat(x) / 640.0,
//                                         y: CGFloat(y) / 640.0))
//            } else {
//                keypoints.append(.zero)
//            }
//        }
//
//        return keypoints
//    }
}
