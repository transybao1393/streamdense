// The ONLY file that knows the engine process + wire protocol. Protocol changes stay here.
//
// Spawn (cap / margin / swap / nice from streamdense.toml [ram] via `streamdense config --json`; defaults shown):
//   sd-watchdog --cap-gb 16 --margin-gb 0 --swap-gb 7 -- nice -n 10 streamdense qwen serve
//       --model DIR --max-ctx 4096 --abort-gb 16 --swap-gb 7 [--mtp]            (cwd = engine/)
// stdin (JSON lines):
//   {"op":"chat","messages":[{"role","content"}],"think":bool,"nnew":int,"mtp":bool,"continue":bool}
//   {"op":"stop"} | {"op":"reset"} | {"op":"stats"}
// stdout (JSON lines): {"ready":true,...} once, {"text":"..."} pieces, then a done record {"done":true,...}
//   ({"error":..} with or without done). Unknown lines are logged. Logs go to stderr (engine + watchdog).
// Spec: docs/SERVE_PROTOCOL.md. Per chat: one {"start":true,n_prompt,kv_reused,n_feed,kv_mode} line, {"text"} pieces,
// exactly one {"done":true,...} line (also after stop / ctx_full / error). Field names also accept the token-level
// `gen` names (prompt_tokens/gen_tokens/continued) so a `gen`-style done record still renders.
import Darwin
import Foundation

struct EngineConfig {
    var root: URL
    var binDir: URL
    var model: URL
    var maxCtx: Int = 4096
    var mtp: Bool = true
    // RAM limits: set in streamdense.toml. Do not raise. Read once at startup (`streamdense config --json`).
    static let ram = RamLimits.load()
    static var capGB: Double { ram.capGB }    // engine cap: sd-watchdog --cap-gb and --abort-gb
    static var swapGB: Double { ram.swapGB }  // watchdog + engine system swap limit

    private static func arg(_ k: String) -> String? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: k), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    static func findRoot() -> URL {
        if let r = ProcessInfo.processInfo.environment["STREAMDENSE_ROOT"] ?? arg("--root") { return URL(fileURLWithPath: r) }
        // walk up from the executable (<root>/app/.build/...) to the first dir that holds engine/
        var d = (Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])).resolvingSymlinksInPath()
        for _ in 0..<8 {
            d = d.deletingLastPathComponent()
            if FileManager.default.fileExists(atPath: d.appendingPathComponent("engine").path) { return d }
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    static func findBinDir(root: URL) -> URL {
        (ProcessInfo.processInfo.environment["STREAMDENSE_BIN"] ?? arg("--bin")).map { URL(fileURLWithPath: $0) }
            ?? root.appendingPathComponent("engine/target/release")
    }

    static func fromEnvironment() -> EngineConfig {
        let env = ProcessInfo.processInfo.environment
        let args = CommandLine.arguments
        let root = findRoot()
        let bin = findBinDir(root: root)
        _ = ram   // load streamdense.toml limits now (once), not on first draw
        let model = (env["STREAMDENSE_MODEL"] ?? arg("--model")).map { URL(fileURLWithPath: $0, relativeTo: root).absoluteURL }
            ?? root.appendingPathComponent("models/qwen38_27b_mlx4")
        var c = EngineConfig(root: root, binDir: bin, model: model)
        if let m = arg("--max-ctx"), let v = Int(m) { c.maxCtx = v }
        if args.contains("--no-mtp") { c.mtp = false }
        return c
    }

    var engineBin: URL { binDir.appendingPathComponent("streamdense") }
    var watchdogBin: URL { binDir.appendingPathComponent("sd-watchdog") }

    var command: [String] {
        let r = EngineConfig.ram
        return [watchdogBin.path, "--cap-gb", gbArg(r.capGB), "--margin-gb", gbArg(r.marginGB), "--swap-gb", gbArg(r.swapGB), "--",
                "nice", "-n", String(r.nice), engineBin.path, "qwen", "serve", "--model", model.path,
                "--max-ctx", String(maxCtx), "--abort-gb", gbArg(r.capGB), "--swap-gb", gbArg(r.swapGB)] + (mtp ? ["--mtp"] : [])
    }
}

private func gbArg(_ v: Double) -> String { v == v.rounded() ? String(Int(v)) : String(v) }

/// [ram] of streamdense.toml, from `streamdense config --json` (run once). Falls back to 16 / 7 / 0 / 10 on any error.
struct RamLimits {
    var capGB = 16.0
    var swapGB = 7.0
    var marginGB = 0.0
    var nice = 10
    var source: String? = nil   // config file path; nil = built-in defaults or fallback
    var error: String? = nil    // why the fallback was used

    static func load() -> RamLimits {
        var r = RamLimits()
        let root = EngineConfig.findRoot()
        let exe = EngineConfig.findBinDir(root: root).appendingPathComponent("streamdense")
        let p = Process()
        p.executableURL = exe
        p.arguments = ["config", "--json"]
        p.currentDirectoryURL = root
        let out = Pipe(), err = Pipe()
        p.standardOutput = out
        p.standardError = err
        do { try p.run() } catch { r.error = "\(exe.path): \(error.localizedDescription)"; return r.logged() }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        let errText = String(data: err.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        p.waitUntilExit()
        guard p.terminationStatus == 0,
              let o = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let ram = o["ram"] as? [String: Any],
              let cap = (ram["cap_gb"] as? NSNumber)?.doubleValue, cap > 0,
              let swap = (ram["swap_gb"] as? NSNumber)?.doubleValue else {
            r.error = "streamdense config --json failed (exit \(p.terminationStatus)): \(errText.trimmingCharacters(in: .whitespacesAndNewlines))"
            return r.logged()
        }
        r.capGB = cap
        r.swapGB = swap
        r.marginGB = (ram["margin_gb"] as? NSNumber)?.doubleValue ?? 0
        r.nice = (ram["nice"] as? NSNumber)?.intValue ?? 10
        r.source = o["source"] as? String
        return r.logged()
    }

    private func logged() -> RamLimits {
        let src = error.map { "fallback (\($0))" } ?? source ?? "built-in defaults"
        FileHandle.standardError.write(Data("[app] RAM limits: cap \(capGB) GB, swap \(swapGB) GB, margin \(marginGB) GB, nice \(nice) from \(src)\n".utf8))
        return self
    }
}

struct ChatTurn { var role: String; var content: String }

/// Done record of one reply, tolerant to field-name variants.
struct DoneRecord {
    var raw: [String: Any]
    private func d(_ keys: String...) -> Double? {
        for k in keys { if let n = raw[k] as? NSNumber { return n.doubleValue } }
        return nil
    }
    private func i(_ keys: String...) -> Int? {
        for k in keys { if let n = raw[k] as? NSNumber { return n.intValue } }
        return nil
    }
    private func b(_ keys: String...) -> Bool? {
        for k in keys { if let n = raw[k] as? Bool { return n } }
        return nil
    }
    var error: String? { raw["error"].map { "\($0)" } }
    var nPrompt: Int? { i("n_prompt", "prompt_tokens") }
    var nNew: Int { i("n_new", "gen_tokens") ?? 0 }
    /// tokens already in the KV that were not run again (chat); gen records only say "continued": bool
    var kvReusedTokens: Int? {
        guard let n = raw["kv_reused"] as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return nil }
        return n.intValue
    }
    var kvReused: Bool {
        if let n = kvReusedTokens { return n > 0 }
        if let n = raw["continued"] as? NSNumber, CFGetTypeID(n) == CFBooleanGetTypeID() { return n.boolValue }
        return false
    }
    var kvMode: String? { raw["kv_mode"] as? String }
    var nFeed: Int? { i("n_feed") }
    var reply: String? { raw["reply"] as? String }
    var prefillMs: Double? { d("prefill_ms") }
    var prefillTokS: Double? {
        if let v = d("prefill_tok_s") { return v }
        if let p = prefillMs, let n = nPrompt, p > 0 { return Double(n) / p * 1e3 }
        return nil
    }
    var ttftMs: Double? { d("ttft_ms") }
    var msPerTok: Double? { d("ms_per_tok") }
    var decodeTokS: Double? { d("decode_tok_s") ?? msPerTok.map { 1e3 / $0 } }
    var gpuMsPerTok: Double? { d("gpu_ms_per_tok") }
    var footprintGB: Double? {
        if let v = d("footprint_gb") { return v }
        if let v = d("footprint") { return v > 1e6 ? v / 1e9 : v }
        return nil
    }
    var peakFootprintGB: Double? { d("peak_footprint_gb", "peak_gb") }
    var stopReason: String { (raw["stop_reason"] as? String) ?? (error != nil ? "error" : "-") }
    var ctxUsed: Int? { i("ctx_used", "pos") }
    var maxCtx: Int? { i("max_ctx") }
    var mtpUsed: Bool { b("mtp") ?? (spec != nil) }
    var mtpRequested: Bool { b("mtp_requested") ?? false }
    /// MTP stats: {"acceptance","tokens_per_pass","passes","accepted"} (nested "spec" or "mtp_stats", or flat mtp_* keys)
    var spec: (acceptance: Double, tokensPerPass: Double?, passes: Int?, accepted: Int?)? {
        if let s = (raw["spec"] ?? raw["mtp_stats"]) as? [String: Any], let a = (s["acceptance"] as? NSNumber)?.doubleValue {
            return (a, (s["tokens_per_pass"] as? NSNumber)?.doubleValue, (s["passes"] as? NSNumber)?.intValue, (s["accepted"] as? NSNumber)?.intValue)
        }
        if let a = d("mtp_acceptance") {
            return (a, d("mtp_tokens_per_pass"), i("mtp_passes"), i("mtp_accepted"))
        }
        return nil
    }
}

enum EngineEvent {
    case ready([String: Any])
    case text(String)
    case token            // token-level serve: a token id without text (counted only)
    case start([String: Any]) // chat: tokenized, prefill starts (n_prompt, kv_reused, n_feed, kv_mode)
    case done(DoneRecord)
    case error(String)
    case info([String: Any]) // stats / reset / other records
    case log(String)
    case exited(code: Int32, reason: String)
}

/// Owns one `sd-watchdog -> nice -> streamdense qwen serve` process tree.
final class EngineClient: @unchecked Sendable {
    let config: EngineConfig
    private var proc: Process?
    private var stdinPipe: Pipe?
    private let writeLock = NSLock()
    private(set) var enginePid: pid_t = 0
    var watchdogPid: pid_t { proc?.processIdentifier ?? 0 }
    var isRunning: Bool { proc?.isRunning ?? false }
    private var logTail: [String] = []
    private let logLock = NSLock()
    private var stopping = false
    var onEvent: ((EngineEvent) -> Void)? // always called on the main queue

    init(config: EngineConfig) { self.config = config }

    func missingBinaries() -> [String] {
        [config.engineBin, config.watchdogBin].filter { !FileManager.default.isExecutableFile(atPath: $0.path) }.map(\.path)
    }

    func start() throws {
        guard proc == nil || !(proc!.isRunning) else { return }
        let cmd = config.command
        let p = Process()
        p.executableURL = URL(fileURLWithPath: cmd[0])
        p.arguments = Array(cmd.dropFirst())
        p.currentDirectoryURL = config.root.appendingPathComponent("engine")
        if !FileManager.default.fileExists(atPath: p.currentDirectoryURL!.path) { p.currentDirectoryURL = config.root }
        let inP = Pipe(), outP = Pipe(), errP = Pipe()
        p.standardInput = inP; p.standardOutput = outP; p.standardError = errP
        stopping = false
        enginePid = 0
        logLock.lock(); logTail.removeAll(); logLock.unlock()
        note("[app] start: " + cmd.joined(separator: " "))
        p.terminationHandler = { [weak self] pr in
            // let the readers drain the last lines first
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.3) { self?.finished(pr) }
        }
        try p.run()
        proc = p; stdinPipe = inP
        readLines(outP.fileHandleForReading) { [weak self] l in self?.handleStdout(l) }
        readLines(errP.fileHandleForReading) { [weak self] l in self?.note(l) }
    }

    private func readLines(_ h: FileHandle, _ cb: @escaping (String) -> Void) {
        let t = Thread {
            var buf = Data()
            while true {
                let chunk = h.availableData
                if chunk.isEmpty { break }
                buf.append(chunk)
                while let nl = buf.firstIndex(of: 0x0A) {
                    let line = String(decoding: buf[buf.startIndex..<nl], as: UTF8.self)
                    buf.removeSubrange(buf.startIndex...nl)
                    cb(line)
                }
            }
            if !buf.isEmpty { cb(String(decoding: buf, as: UTF8.self)) }
        }
        t.start()
    }

    private func emit(_ e: EngineEvent) { DispatchQueue.main.async { self.onEvent?(e) } }

    private func note(_ line: String) {
        let l = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !l.isEmpty else { return }
        logLock.lock(); logTail.append(l); if logTail.count > 300 { logTail.removeFirst(logTail.count - 300) }; logLock.unlock()
        emit(.log(l))
    }

    private func handleStdout(_ line: String) {
        let s = line.trimmingCharacters(in: .whitespaces)
        guard !s.isEmpty else { return }
        guard let data = s.data(using: .utf8), let v = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            note("[app] non-JSON stdout: " + s); return
        }
        if (v["ready"] as? Bool) == true {
            if let pid = (v["pid"] as? NSNumber)?.int32Value { enginePid = pid }
            emit(.ready(v))
        } else if (v["start"] as? Bool) == true {
            emit(.start(v))
        } else if let t = v["text"] as? String {
            emit(.text(t))
            if (v["done"] as? Bool) == true { emit(.done(DoneRecord(raw: v))) }
        } else if (v["done"] as? Bool) == true {
            emit(.done(DoneRecord(raw: v)))
        } else if let e = v["error"] {
            emit(.error("\(e)"))
        } else if v["tok"] != nil {
            emit(.token)
        } else {
            emit(.info(v))
        }
    }

    private func finished(_ pr: Process) {
        if let cur = proc, cur !== pr { return } // an older process tree (unloaded, then loaded again)
        let code = pr.terminationStatus
        logLock.lock(); let log = logTail; logLock.unlock()
        let tail = log.suffix(40).joined(separator: "\n")
        var reason: String
        if stopping {
            reason = ""
        } else if tail.contains("[sd-watchdog]") && tail.contains("KILL") {
            reason = log.last(where: { $0.contains("KILL") }) ?? "killed by the watchdog"
        } else if tail.contains("[sd-watchdog] exit None") {
            reason = "engine killed by a signal (not by the watchdog)"
        } else if tail.contains("FOOTPRINT MONITOR") {
            reason = log.last(where: { $0.contains("FOOTPRINT MONITOR") }) ?? "footprint monitor abort"
        } else if tail.contains("ERROR") {
            reason = log.last(where: { $0.contains("ERROR") }) ?? "error"
        } else {
            reason = "engine exited with code \(code)"
        }
        if !reason.isEmpty { reason += " (exit code \(code))" }
        enginePid = 0
        emit(.exited(code: code, reason: reason))
    }

    /// Engine pid: from the ready line, else the watchdog's child (during loading).
    func currentEnginePid() -> pid_t {
        if enginePid > 0 { return enginePid }
        guard let p = proc, p.isRunning else { return 0 }
        return SystemStats.childPids(p.processIdentifier).first ?? 0
    }

    // ---------------------------------------------------------------- requests
    private func send(_ obj: [String: Any]) -> Bool {
        guard let h = stdinPipe?.fileHandleForWriting, isRunning,
              var d = try? JSONSerialization.data(withJSONObject: obj) else { return false }
        d.append(0x0A)
        writeLock.lock(); defer { writeLock.unlock() }
        do { try h.write(contentsOf: d); return true } catch { return false }
    }

    @discardableResult
    func chat(messages: [ChatTurn], think: Bool, reasoningEffort: String?, nnew: Int, mtp: Bool, keepKV: Bool) -> Bool {
        var o: [String: Any] = ["op": "chat", "messages": messages.map { ["role": $0.role, "content": $0.content] },
                                "think": think, "nnew": nnew, "mtp": mtp, "continue": keepKV]
        if think, let r = reasoningEffort { o["reasoning_effort"] = r }
        return send(o)
    }
    @discardableResult func stop() -> Bool { send(["op": "stop"]) }
    @discardableResult func reset() -> Bool { send(["op": "reset"]) }
    @discardableResult func stats() -> Bool { send(["op": "stats"]) }

    /// Clean shutdown: stop, close stdin (engine exits on EOF, watchdog follows), then escalate.
    /// Blocks up to `timeout` seconds. Safe to call from applicationWillTerminate.
    func shutdown(timeout: Double = 15) {
        guard let p = proc else { return }
        stopping = true
        let ePid = currentEnginePid()
        if p.isRunning {
            _ = stop()
            try? stdinPipe?.fileHandleForWriting.close()
            let t0 = Date()
            while p.isRunning && Date().timeIntervalSince(t0) < timeout { usleep(50_000) }
            if p.isRunning || SystemStats.isAlive(ePid) {
                // the watchdog does not forward signals: kill the engine first, the watchdog then exits by itself
                if ePid > 0 { note("[app] engine did not exit in \(Int(timeout)) s: SIGKILL \(ePid)"); kill(ePid, SIGKILL) }
                let t1 = Date()
                while p.isRunning && Date().timeIntervalSince(t1) < 3 { usleep(50_000) }
                if p.isRunning { p.terminate() }
                let t2 = Date()
                while p.isRunning && Date().timeIntervalSince(t2) < 2 { usleep(50_000) }
                if p.isRunning { kill(p.processIdentifier, SIGKILL) }
            }
        }
        if SystemStats.isAlive(ePid) { kill(ePid, SIGKILL) }
        proc = nil; stdinPipe = nil; enginePid = 0
    }
}
