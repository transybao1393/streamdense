// Entry point. `--print-stats` prints one system sample and the engine command, then exits (no window, no engine).
import AppKit
import Darwin
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    var model: ChatModel?
    private var signalSources: [DispatchSourceSignal] = []

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular) // plain executable (no .app bundle): still a normal Dock app with a window
        NSApp.activate(ignoringOtherApps: true)
        // Ctrl+C / kill from run.sh: free the engine first, then exit
        for sig in [SIGINT, SIGTERM, SIGHUP] {
            signal(sig, SIG_IGN)
            let s = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            s.setEventHandler { NSApp.terminate(nil) }
            s.resume()
            signalSources.append(s)
        }
        // --snapshot FILE.png: render the main window in-process after 2.5 s, then quit (screencapture may be blocked)
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--snapshot"), i + 1 < args.count {
            let path = args[i + 1]
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                if let v = NSApp.windows.first(where: { $0.isVisible && $0.contentView != nil })?.contentView,
                   let rep = v.bitmapImageRepForCachingDisplay(in: v.bounds) {
                    v.cacheDisplay(in: v.bounds, to: rep)
                    try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
                    print("snapshot: \(path)")
                } else {
                    print("snapshot: no window")
                }
                NSApp.terminate(nil)
            }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated { model?.shutdownForQuit() }
    }
}

@main
struct StreamdenseChatApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var model: ChatModel

    init() {
        if CommandLine.arguments.contains("--print-stats") {
            StreamdenseChatApp.printStats()
            exit(0)
        }
        _model = StateObject(wrappedValue: ChatModel())
    }

    var body: some Scene {
        WindowGroup("streamdense chat") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 960, minHeight: 620)
                .onAppear { delegate.model = model }
        }
        .defaultSize(width: 1400, height: 900)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }

    static func printStats() {
        let c = EngineConfig.fromEnvironment()
        let vm = SystemStats.vm()
        let me = SystemStats.pidUsage(getpid())
        print(String(format: "available %.2f GB, compressor %.2f GB, swap %.2f GB", vm?.availGB ?? -1, vm?.compressorGB ?? -1, vm?.swapGB ?? -1))
        print("gpu util %: \(SystemStats.gpuUtil().map(String.init) ?? "n/a")")
        print(String(format: "self cpu %.3f s, footprint %.3f GB", me?.cpuS ?? -1, Double(me?.footprint ?? 0) / 1e9))
        let g = SystemStats.ramGate()
        print("RAM gate: \(g.msg) -> \(g.ok ? "GO" : "STOP")")
        print("root: \(c.root.path)")
        print("engine cmd: \(c.command.joined(separator: " "))")
    }
}
