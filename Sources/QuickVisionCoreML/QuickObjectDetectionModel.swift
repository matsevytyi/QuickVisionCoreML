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
    private var outputConfName: String = "coordinates"
    private var outputLocationName: String = "output"
    private var outputLabelName: String? // not for YOLO - only if class
    
    private var outputConfShape: [Int]
    private var outputLocationShape: [Int]
    private var outputLabelShape: [Int]

    // for YOLO
// named 'confidence', MultiArray (Float32 0 × 80), [0...x80] Boxes x Class confidence (see user-defined metadata "classes")
// named 'coordinates', MultiArray (Float32 0 × 4), [0...x4] Boxes × [x, y, width, height] (relative to image size)
    
    // for DETR
// named 'boxes', MultiArray (Float16 1 × 300 × 4)
// named 'scores', MultiArray (Float16 1 × 300)
// named 'label', MultiArray (Float16 1 × 300)
    
    private let classLabels: [Any]?
    
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
        
        
        print(outputs[self.outputConfName])
        print(outputs[self.outputConfName]?.multiArrayConstraint)
        print(outputs[self.outputConfName]?.multiArrayConstraint?.shape)
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

}

