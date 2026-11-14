// swift-tools-version:5.10
// streamdense chat: native macOS SwiftUI chat for the Qwen engine (`qwen serve`). Build: swift build -c release
import PackageDescription

let package = Package(
    name: "StreamdenseChat",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "StreamdenseChat",
            path: "Sources/StreamdenseChat",
            linkerSettings: [.linkedFramework("IOKit"), .linkedFramework("AppKit")]
        )
    ]
)
