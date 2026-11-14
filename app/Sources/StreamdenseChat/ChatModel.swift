// App state: conversation, engine lifecycle, per-reply metrics, 1 Hz system samples.
import Darwin
import Foundation
import SwiftUI

enum EngineState: String {
    case unloaded, loading, ready, busy, unloading, dead
}

struct Message: Identifiable {
    let id = UUID()
    var role: String          // "user" | "assistant"
    var text: String
    var streaming = false
    var inContext = true      // false: dropped from the conversation (error / nothing generated)
    var thinkingOn = false
    var done: DoneRecord?
    var replyNo: Int?
    var note: String?         // status line under the bubble
    var error: String?
}

struct ReplyRow: Identifiable {
    let id = UUID()
    var n: Int
    var done: DoneRecord
    var uiTtftMs: Double?
    var uiTotalMs: Double
    var tokensSeen: Int
    var mtpSpeedup: Double?
    var mtpSpeedupRef: Int?
    var enginePeakGB: Double?
}

struct Sample {
    var t = Date()
    var vm: VMStats?
    var engineFpGB: Double?
    var engineCpu: Double?
    var appCpu: Double?
    var appFpGB: Double?
    var gpu: Int?
}

@MainActor
final class ChatModel: ObservableObject {
    let engine: EngineClient
    var config: EngineConfig { engine.config }

    @Published var state: EngineState = .unloaded
    @Published var deathReason = ""
    @Published var messages: [Message] = []
    @Published var input = ""
    @Published var maxNew = 512
    @Published var think = false
    @Published var keepKV = true
    @Published var mtp = true
    @Published var reasoningEffort = "xhigh"   // only sent with think:true (xhigh | medium | low)
    @Published var chatAvailable: Bool?
    @Published var thinkingLocked: Bool?   // fixed for the conversation at its first turn (same as the web UI)
    @Published var status = ""
    @Published var replies: [ReplyRow] = []
    @Published var sample = Sample()
    @Published var ramHist: [Double?] = []
    @Published var peakFpGB: Double = 0
    @Published var log: [String] = []
    @Published var readyInfo: [String: Any] = [:]
    @Published var loadSeconds: Double?
    @Published var mtpAvailable: Bool?
    @Published var watchdogLast = ""

    private var loadStart: Date?
    private var pendingSend: String?
    private var streamBuf = ""
    private var flushScheduled = false
    private var reqStart: Date?
    private var firstText: Date?
    private var tokensSeen = 0
    private var plainRef: (ms: Double, n: Int)?
    private var cpuPrev: [String: (Date, Double)] = [:]
    private var timer: Timer?
    private let sampleQ = DispatchQueue(label: "sampler", qos: .utility)

    init(config: EngineConfig = .fromEnvironment()) {
        engine = EngineClient(config: config)
        mtp = config.mtp
        engine.onEvent = { [weak self] e in
            MainActor.assumeIsolated { self?.handle(e) }
        }
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
        tick()
        // --autoload: load the model at startup (RAM gate first, as for "Load model").
        // --autosend TEXT (debug): send TEXT as the first message (loads the model if needed); the done line is logged.
        let args = CommandLine.arguments
        let autosend = args.firstIndex(of: "--autosend").flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }
        if autosend != nil || args.contains("--autoload") {
            DispatchQueue.main.async { [weak self] in
                MainActor.assumeIsolated {
                    guard let self else { return }
                    if let t = autosend { self.input = t; self.send() } else { self.load() }
                }
            }
        }
    }

    var loaded: Bool { [.ready, .busy, .loading].contains(state) }
    var busy: Bool { state == .busy }
    var lastReply: ReplyRow? { replies.last }
    var turns: Int { messages.filter { $0.role == "user" && $0.inContext && !$0.streaming }.count }
    @Published var kvKept = false

    // ---------------------------------------------------------------- lifecycle
    func load() {
        guard !loaded, state != .unloading else { return }
        let miss = engine.missingBinaries()
        if !miss.isEmpty {
            status = "missing \(miss.joined(separator: ", ")): run 'cargo build --release --offline' in engine/"
            appendLog("[app] " + status); return
        }
        guard FileManager.default.fileExists(atPath: config.model.path) else {
            status = "model dir not found: \(config.model.path)"; appendLog("[app] " + status); return
        }
        let g = SystemStats.ramGate()
        appendLog("[app] RAM gate: \(g.msg) -> \(g.ok ? "GO" : "STOP")")
        guard g.ok else {
            status = "RAM gate STOP: \(g.msg)"
            pendingSend = nil
            return
        }
        do {
            deathReason = ""
            readyInfo = [:]; loadSeconds = nil; peakFpGB = 0; kvKept = false
            state = .loading
            status = "loading model..."
            loadStart = Date()
            try engine.start()
        } catch {
            state = .dead
            deathReason = "failed to start the engine: \(error.localizedDescription)"
            status = deathReason
        }
    }

    func unload() {
        guard loaded else { return }
        if busy { engine.stop() }
        state = .unloading
        status = "unloading..."
        kvKept = false
        let eng = engine
        DispatchQueue.global().async {
            eng.shutdown()
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    self.finishStreamingMessage(error: "stopped: model unloaded")
                    self.state = .unloaded
                    self.status = "model unloaded"
                }
            }
        }
    }

    func restart() {
        clearConversation()
        state = .unloaded
        load()
    }

    /// Called on quit: blocks until the engine and its watchdog are gone.
    func shutdownForQuit() {
        timer?.invalidate()
        engine.shutdown()
    }

    // ---------------------------------------------------------------- chat
    func send() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !busy, state != .unloading, state != .loading else { return }
        input = ""
        if state != .ready {
            pendingSend = text
            messages.append(Message(role: "user", text: text, note: "waiting for the model to load"))
            load()
            if state != .loading {   // gate STOP or start failure
                if let k = messages.indices.last { messages[k].inContext = false; messages[k].note = nil; messages[k].error = status }
                pendingSend = nil
            }
            return
        }
        messages.append(Message(role: "user", text: text))
        startReply()
    }

    private func startReply() {
        if thinkingLocked == nil { thinkingLocked = think }
        let th = thinkingLocked ?? false
        let convo = messages.filter { $0.inContext && !$0.streaming }.map { ChatTurn(role: $0.role, content: $0.text) }
        if let k = messages.indices.last { messages[k].note = nil }
        let useMtp = mtp && (mtpAvailable ?? config.mtp)
        var m = Message(role: "assistant", text: "", streaming: true, thinkingOn: th)
        m.note = keepKV && kvKept ? "continuing on the kept KV..." : "prefill..."
        messages.append(m)
        streamBuf = ""; tokensSeen = 0; reqStart = Date(); firstText = nil
        state = .busy
        status = ""
        if !engine.chat(messages: convo, think: th, reasoningEffort: reasoningEffort, nnew: max(1, maxNew), mtp: useMtp, keepKV: keepKV) {
            finishStreamingMessage(error: "could not write to the engine")
            state = engine.isRunning ? .ready : .dead
        }
    }

    func stop() { if busy { engine.stop() } }

    func clearConversation() {
        guard !busy else { status = "stop the reply first"; return }
        messages.removeAll()
        thinkingLocked = nil
        kvKept = false
        if state == .ready { engine.reset() }
    }

    // ---------------------------------------------------------------- engine events
    private func handle(_ e: EngineEvent) {
        switch e {
        case .ready(let info):
            readyInfo = info
            mtpAvailable = info["mtp_available"] as? Bool
            chatAvailable = info["chat_available"] as? Bool
            if chatAvailable == false {
                let why = (info["tokenizer_error"] as? String) ?? "tokenizer did not load"
                appendLog("[engine] chat unavailable: " + why)
            }
            loadSeconds = loadStart.map { Date().timeIntervalSince($0) }
            state = .ready
            status = String(format: "loaded in %.1f s", loadSeconds ?? 0)
            if chatAvailable == false { status += " · chat UNAVAILABLE (tokenizer did not load, see log)" }
            appendLog(String(format: "[app] engine ready in %.1f s, footprint %.2f GB, chat %@, mtp %@", loadSeconds ?? 0,
                             (info["footprint_gb"] as? NSNumber)?.doubleValue ?? -1, chatAvailable == true ? "yes" : "no",
                             mtpAvailable == true ? "yes" : "no"))
            if let p = pendingSend {
                pendingSend = nil
                _ = p
                startReply()
            }
        case .start(let v):
            if let k = messages.lastIndex(where: { $0.streaming }) {
                let n = (v["n_prompt"] as? NSNumber)?.intValue ?? 0
                let reused = (v["kv_reused"] as? NSNumber)?.intValue ?? 0
                let feed = (v["n_feed"] as? NSNumber)?.intValue ?? n
                let mode = (v["kv_mode"] as? String) ?? "?"
                messages[k].note = mode == "full" ? "full prefill of \(n) tokens..."
                    : "prefill \(feed) new tokens (\(reused) reused from the KV, \(mode))..."
            }
        case .text(let t):
            if firstText == nil { firstText = Date() }
            streamBuf += t
            scheduleFlush()
        case .token:
            tokensSeen += 1
            if firstText == nil { firstText = Date() }
        case .done(let d):
            flush()
            finishReply(d)
        case .error(let msg):
            appendLog("[engine] error: " + msg)
            if busy { finishStreamingMessage(error: msg); state = .ready }
            else { status = msg }
        case .info(let v):
            if v["stats"] != nil { appendLog("[engine] stats: \(v)") }
        case .log(let l):
            appendLog(l)
            if l.contains("[sd-watchdog]") { watchdogLast = l }
        case .exited(_, let reason):
            if state == .unloading || reason.isEmpty {
                state = .unloaded
            } else {
                state = .dead
                deathReason = reason
                status = "engine stopped: " + reason
                finishStreamingMessage(error: "engine died: " + reason)
            }
            kvKept = false
            if pendingSend != nil, let k = messages.lastIndex(where: { $0.role == "user" }) {
                messages[k].inContext = false; messages[k].note = nil; messages[k].error = "not sent: the engine stopped while loading"
            }
            pendingSend = nil
        }
    }

    private func scheduleFlush() {
        guard !flushScheduled else { return }
        flushScheduled = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) { [weak self] in
            MainActor.assumeIsolated { self?.flush() }
        }
    }

    private func flush() {
        flushScheduled = false
        guard let k = messages.lastIndex(where: { $0.streaming }) else { return }
        if messages[k].text != streamBuf { messages[k].text = streamBuf }
        if !streamBuf.isEmpty, messages[k].note != nil, messages[k].note != "generating..." { messages[k].note = "generating..." }
    }

    private func finishStreamingMessage(error: String) {
        guard let k = messages.lastIndex(where: { $0.streaming }) else { return }
        flush()
        messages[k].streaming = false
        messages[k].inContext = false
        messages[k].error = error
        messages[k].note = nil
        if k > 0, messages[k - 1].role == "user" { messages[k - 1].inContext = false }
        kvKept = false
    }

    private func finishReply(_ d: DoneRecord) {
        let now = Date()
        state = engine.isRunning ? .ready : state
        guard let k = messages.lastIndex(where: { $0.streaming }) else { return }
        if let err = d.error, d.stopReason != "ctx_full" {
            finishStreamingMessage(error: err)
            return
        }
        messages[k].streaming = false
        messages[k].done = d
        messages[k].note = nil
        // the spec asks for the reply exactly as received (text pieces concatenated, or done.reply)
        var text = d.reply ?? streamBuf
        for s in ["<|im_end|>", "<|endoftext|>"] where text.hasSuffix(s) { text.removeLast(s.count) }
        messages[k].text = text
        if d.nNew == 0 {
            messages[k].inContext = false
            if k > 0, messages[k - 1].role == "user" { messages[k - 1].inContext = false }
        }
        if d.stopReason == "ctx_full" {
            messages[k].error = "context is full (max_ctx \(d.maxCtx ?? config.maxCtx)): press Clear to start a new conversation"
        }
        kvKept = d.stopReason != "ctx_full" && d.nNew > 0
        var row = ReplyRow(n: replies.count + 1, done: d,
                           uiTtftMs: firstText.flatMap { f in reqStart.map { f.timeIntervalSince($0) * 1e3 } },
                           uiTotalMs: reqStart.map { now.timeIntervalSince($0) * 1e3 } ?? 0,
                           tokensSeen: tokensSeen, enginePeakGB: peakFpGB > 0 ? peakFpGB : nil)
        if let ms = d.msPerTok, d.nNew >= 9 {
            if d.mtpUsed {
                if let p = plainRef { row.mtpSpeedup = p.ms / ms; row.mtpSpeedupRef = p.n }
            } else {
                plainRef = (ms, row.n)
            }
        }
        messages[k].replyNo = row.n
        replies.append(row)
        appendLog(String(format: "[app] reply %d done: %@, %d tokens, prefill %.1f tok/s, decode %.1f tok/s, kv %@, footprint %.2f GB",
                         row.n, d.stopReason, d.nNew, (d.raw["prefill_tok_s"] as? NSNumber)?.doubleValue ?? -1,
                         (d.raw["decode_tok_s"] as? NSNumber)?.doubleValue ?? -1, (d.raw["kv_mode"] as? String) ?? "?",
                         (d.raw["footprint_gb"] as? NSNumber)?.doubleValue ?? -1))
    }

    // ---------------------------------------------------------------- samples
    private func appendLog(_ l: String) {
        print(l); fflush(stdout)   // also to the terminal / run.sh log
        log.append(l)
        if log.count > 300 { log.removeFirst(log.count - 300) }
    }

    private func tick() {
        let pid = engine.currentEnginePid()
        sampleQ.async { [weak self] in
            var s = Sample()
            s.vm = SystemStats.vm()
            s.gpu = SystemStats.gpuUtil()
            let eu = SystemStats.pidUsage(pid)
            let au = SystemStats.pidUsage(getpid())
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    guard let self else { return }
                    if let eu {
                        s.engineFpGB = Double(eu.footprint) / 1e9
                        s.engineCpu = self.cpuPct("eng", eu.cpuS, s.t)
                        self.peakFpGB = max(self.peakFpGB, s.engineFpGB ?? 0)
                    } else {
                        self.cpuPrev.removeValue(forKey: "eng")
                    }
                    if let au {
                        s.appCpu = self.cpuPct("app", au.cpuS, s.t)
                        s.appFpGB = Double(au.footprint) / 1e9
                    }
                    self.sample = s
                    self.ramHist.append(s.engineFpGB)
                    if self.ramHist.count > 600 { self.ramHist.removeFirst(self.ramHist.count - 600) }
                }
            }
        }
    }

    private func cpuPct(_ key: String, _ cpu: Double, _ t: Date) -> Double? {
        defer { cpuPrev[key] = (t, cpu) }
        guard let p = cpuPrev[key] else { return nil }
        return max(0, 100 * (cpu - p.1) / max(1e-6, t.timeIntervalSince(p.0)))
    }
}
