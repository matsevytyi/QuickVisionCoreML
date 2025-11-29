// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "QuickVisionCoreML",
    platforms: [
            .iOS(.v16)        // choose your minimum iOS version
        ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "QuickVisionCoreML",
            targets: ["QuickVisionCoreML"])
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "QuickVisionCoreML"),
        .testTarget(
            name: "QuickVisionCoreMLTests",
            dependencies: ["QuickVisionCoreML"]
        )
    ]
)
