//
//  QuickObjectClassificationModel.swift
//  QuickVisionCoreML
//
//  Created by Andrii Matsevytyi on 28.11.2025.
//
import Foundation
import CoreML
import Accelerate
import CoreImage

public class QuickObjectClassificationModel {
    
    // MARK: Properties

    private let model: MLModel

    // Input features
    private var inputName: String
    private var inputWidth: Int = 640
    private var inputHeight: Int = 640

    // Output features
    private var labelOutputName: String?      // e.g. "classLabel"
    private var probOutputName: String?       // e.g. "classLabelProbs" or logits
    private var classLabels: [String]?

    // MARK: Init

    public init(model: MLModel) throws {
        self.model = model

        let desc = model.modelDescription

        // Input selection: prefer image input
        let inputs = desc.inputDescriptionsByName
        if let (name, feature) = inputs.first,
           feature.type == .image,
           let constraint = feature.imageConstraint {
            self.inputName = name
            self.inputWidth = constraint.pixelsWide
            self.inputHeight = constraint.pixelsHigh
        } else {
            self.inputName = inputs.keys.first ?? "image"
            self.inputWidth = inputs[self.inputName]?.imageConstraint?.pixelsWide ?? 640
            self.inputHeight = inputs[self.inputName]?.imageConstraint?.pixelsHigh ?? 640
        }

        // Outputs: try to find label (string) and probabilities (dictionary or multiArray)
        let outputs = desc.outputDescriptionsByName

        // string label output (standard CoreML image classifier)
        if let labelEntry = outputs.first(where: { $0.value.type == .string }) {
            self.labelOutputName = labelEntry.key
        } else {
            self.labelOutputName = nil
        }

        // probabilities / logits output
        if let probEntry = outputs.first(where: { $0.value.type == .dictionary || $0.value.type == .multiArray }) {
            self.probOutputName = probEntry.key
        } else {
            self.probOutputName = nil
        }

        // Class labels (if present)
        if let labels = desc.classLabels as? [String] {
            self.classLabels = labels
        } else {
            self.classLabels = nil
        }
    }


    public convenience init(model: MLModel, config: [String: Any]) throws {
        try self.init(model: model)

        if let inputName = config["inputName"] as? String {
            self.inputName = inputName
        }

        if let inputWidth = config["inputWidth"] as? Int {
            self.inputWidth = inputWidth
        }

        if let inputHeight = config["inputHeight"] as? Int {
            self.inputHeight = inputHeight
        }

        if let labelOutputName = config["labelOutputName"] as? String {
            self.labelOutputName = labelOutputName
        }

        if let probOutputName = config["probOutputName"] as? String {
            self.probOutputName = probOutputName
        }

        if let classLabels = config["classLabels"] as? [String] {
            self.classLabels = classLabels
        }
    }

    // MARK: Public API (2 functions)

    public func predict(image: CGImage) -> String? {
        do {
            let pixelBuffer = try preprocessImage(image)
            return try predictHelper(pixelBuffer: pixelBuffer)
        } catch {
            print("[QuickObjectClassificationModel] predict(image:) error:", error)
            return nil
        }
    }

    public func predict(pixelBuffer: CVPixelBuffer) -> String? {
        do {
            let resized = try resizePixelBuffer(pixelBuffer)
            return try predictHelper(pixelBuffer: resized)
        } catch {
            print("[QuickObjectClassificationModel] predict(pixelBuffer:) error:", error)
            return nil
        }
    }

    // MARK: Core implementation

    private func predictHelper(pixelBuffer: CVPixelBuffer) throws -> String? {
        let inputValue = MLFeatureValue(pixelBuffer: pixelBuffer)
        let input = try MLDictionaryFeatureProvider(dictionary: [inputName: inputValue])

        let prediction = try model.prediction(from: input)

        // 1) If model already gives a string label: use it
        if let labelName = labelOutputName,
           let labelValue = prediction.featureValue(for: labelName)?.stringValue {
            return labelValue
        }

        // 2) Otherwise, try to derive label from probabilities/logits and classLabels
        if let probName = probOutputName {
            if let dict = prediction.featureValue(for: probName)?.dictionaryValue {
                return parseTopFromDictionary(dict)
            } else if let array = prediction.featureValue(for: probName)?.multiArrayValue {
                return parseTopFromMultiArray(array)
            }
        }

        print("[QuickObjectClassificationModel] No suitable output found for label")
        return nil
    }

    // MARK: Parsing helpers

    private func parseTopFromDictionary(_ dict: [AnyHashable: NSNumber]) -> String? {
        // Keys can be String or Int; we assume String for class names
        var bestKey: String?
        var bestScore: Double = -Double.infinity

        for (key, value) in dict {
            guard let name = key as? String else { continue }
            let score = value.doubleValue
            if score > bestScore {
                bestScore = score
                bestKey = name
            }
        }
        return bestKey
    }

    private func parseTopFromMultiArray(_ array: MLMultiArray) -> String? {
        // Assume logits/probs over classLabels (and same length)
        guard let labels = classLabels else {
            print("[QuickObjectClassificationModel] classLabels missing for multiArray output")
            return nil
        }
        guard array.count == labels.count else {
            print("[QuickObjectClassificationModel] classLabels count (\(labels.count)) != logits count (\(array.count))")
            return nil
        }

        var bestIndex = 0
        var bestScore = -Double.infinity

        for i in 0..<array.count {
            let s = array[i].doubleValue
            if s > bestScore {
                bestScore = s
                bestIndex = i
            }
        }

        return labels[bestIndex]
    }

    // MARK: Preprocessing

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
            throw NSError(
                domain: "PixelBufferResize",
                code: 0,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create pixel buffer"]
            )
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
            throw NSError(
                domain: "ImageProcessing",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create pixel buffer"]
            )
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
            throw NSError(
                domain: "ImageProcessing",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create CGContext"]
            )
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: inputWidth, height: inputHeight))

        return buffer
    }

}
