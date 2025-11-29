import Foundation
import CoreML


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

