// Main window: header, death banner, chat (hero) + always-visible input, side panel.
import AppKit
import SwiftUI

enum Theme {
    static let bg = Color(red: 0.043, green: 0.055, blue: 0.075)
    static let panel = Color(red: 0.078, green: 0.098, blue: 0.133)
    static let panel2 = Color(red: 0.106, green: 0.133, blue: 0.188)
    static let line = Color(red: 0.173, green: 0.212, blue: 0.278)
    static let accent = Color(red: 0.373, green: 0.702, blue: 1.0)
    static let good = Color(red: 0.29, green: 0.87, blue: 0.5)
    static let warn = Color(red: 0.98, green: 0.75, blue: 0.14)
    static let bad = Color(red: 1.0, green: 0.36, blue: 0.36)
    static let user = Color(red: 0.114, green: 0.208, blue: 0.341)
    static let bot = Color(red: 0.094, green: 0.125, blue: 0.173)
}

func fmt(_ x: Double?, _ d: Int = 1) -> String {
    guard let x, x.isFinite else { return "-" }
    return String(format: "%.\(d)f", x)
}

struct ContentView: View {
    @EnvironmentObject var m: ChatModel
    var body: some View {
        VStack(spacing: 0) {
            HeaderBar()
            if m.state == .dead { DeathBanner() }
            HSplitView {
                ChatColumn().frame(minWidth: 480)
                SidePanel().frame(minWidth: 340, idealWidth: 420, maxWidth: 560)
            }
        }
        .background(Theme.bg)
        .preferredColorScheme(.dark)
    }
}

struct StateBadge: View {
    let state: EngineState
    var body: some View {
        let (bg, fg): (Color, Color) = {
            switch state {
            case .ready: return (Theme.good, .black)
            case .busy, .loading, .unloading: return (Theme.warn, .black)
            case .dead: return (Theme.bad, .white)
            case .unloaded: return (Color(white: 0.82), .black)
            }
        }()
        Text(state.rawValue).font(.system(size: 12, weight: .semibold)).foregroundStyle(fg)
            .padding(.horizontal, 9).padding(.vertical, 3).background(Capsule().fill(bg))
    }
}

struct HeaderBar: View {
    @EnvironmentObject var m: ChatModel
    var body: some View {
        HStack(spacing: 12) {
            Text("streamdense · Qwen chat").font(.system(size: 16, weight: .semibold)).foregroundStyle(.primary)
            StateBadge(state: m.state)
            Text("\(m.config.model.lastPathComponent) · max_ctx \(String(m.config.maxCtx))").foregroundStyle(.secondary)
            Spacer()
            Button("Load model") { m.load() }
                .buttonStyle(.borderedProminent)
                .disabled(m.loaded || m.state == .unloading)
            Button("Unload (free RAM)") { m.unload() }
                .disabled(!m.loaded)
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
        .background(Theme.panel)
        .overlay(alignment: .bottom) { Rectangle().fill(Theme.line).frame(height: 1) }
    }
}

struct DeathBanner: View {
    @EnvironmentObject var m: ChatModel
    var body: some View {
        HStack(spacing: 12) {
            Text("Engine stopped:").bold().foregroundStyle(.white)
            Text(m.deathReason).foregroundStyle(.white).textSelection(.enabled).lineLimit(3)
            Spacer()
            Button("Restart") { m.restart() }.buttonStyle(.borderedProminent)
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
        .background(Color(red: 0.35, green: 0.07, blue: 0.07))
        .overlay(alignment: .bottom) { Rectangle().fill(Theme.bad).frame(height: 2) }
    }
}

struct ChatColumn: View {
    @EnvironmentObject var m: ChatModel
    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        if m.messages.isEmpty {
                            Text(m.loaded ? "Model loaded. Type a message below." : "No model loaded. Load it with \"Load model\" or just send a message (the RAM gate runs first).")
                                .foregroundStyle(.secondary).padding(.top, 20)
                        }
                        ForEach(m.messages) { msg in
                            MessageBubble(msg: msg).id(msg.id)
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding(16)
                }
                .onChange(of: m.messages.last?.text) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
                .onChange(of: m.messages.count) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
            }
            InputBar()
        }
        .background(Theme.bg)
    }
}

struct MessageBubble: View {
    let msg: Message
    @State private var thinkOpen = true
    var body: some View {
        HStack {
            if msg.role == "user" { Spacer(minLength: 60) }
            VStack(alignment: .leading, spacing: 6) {
                if msg.role == "user" {
                    MarkdownView(text: msg.text)
                } else {
                    let sp = ThinkSplit.split(msg.text, thinkingOn: msg.thinkingOn)
                    if let th = sp.thinking {
                        VStack(alignment: .leading, spacing: 4) {
                            Button {
                                thinkOpen.toggle()
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: thinkOpen ? "chevron.down" : "chevron.right")
                                    Text(sp.thinkingOpen ? "THINKING..." : "THINKING").font(.system(size: 11, weight: .bold))
                                    if !thinkOpen { Text("(\(th.count) chars)").font(.system(size: 11)).foregroundStyle(.secondary) }
                                }.foregroundStyle(Theme.accent)
                            }.buttonStyle(.plain)
                            if thinkOpen {
                                MarkdownView(text: th).foregroundStyle(.secondary)
                            }
                        }
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(Color(red: 0.067, green: 0.094, blue: 0.141))
                        .overlay(alignment: .leading) { Rectangle().fill(Theme.accent).frame(width: 3) }
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                    }
                    if !sp.answer.isEmpty { MarkdownView(text: sp.answer) }
                    if msg.streaming && msg.text.isEmpty { ProgressView().controlSize(.small) }
                }
                if let n = msg.note {
                    Text(n).font(.caption).foregroundStyle(Theme.accent)
                }
                if let d = msg.done {
                    Divider().overlay(Theme.line)
                    Text(metaText(d, n: msg.replyNo)).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                }
                if let e = msg.error {
                    Text(e).font(.caption.weight(.bold)).foregroundStyle(e.hasPrefix("context is full") ? Theme.warn : Theme.bad)
                        .textSelection(.enabled)
                }
            }
            .textSelection(.enabled)
            .padding(.horizontal, 13).padding(.vertical, 10)
            .background(RoundedRectangle(cornerRadius: 10).fill(msg.role == "user" ? Theme.user : Theme.bot))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(msg.role == "user" ? Color.clear : Theme.line))
            .opacity(msg.inContext || msg.streaming ? 1 : 0.85)
            if msg.role != "user" { Spacer(minLength: 60) }
        }
    }

    func metaText(_ d: DoneRecord, n: Int?) -> String {
        var s = "\(d.nNew) tok · \(fmt(d.decodeTokS, 2)) tok/s · prefill \(d.nPrompt.map(String.init) ?? "-") tok in \(fmt(d.prefillMs, 0)) ms (\(fmt(d.prefillTokS, 1)) tok/s) · TTFT \(fmt(d.ttftMs, 0)) ms · \(d.stopReason)"
        if let r = d.kvReusedTokens, r > 0 { s += " · \(r) tok reused from KV (\(d.kvMode ?? "-"))" }
        else if d.kvReused { s += " · KV reused" }
        if let sp = d.spec {
            s += " · MTP \(fmt(sp.acceptance * 100, 0))% accepted"
            if let t = sp.tokensPerPass { s += ", \(fmt(t, 2)) tok/pass" }
        }
        return s
    }
}

struct InputBar: View {
    @EnvironmentObject var m: ChatModel
    @FocusState private var focused: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ZStack(alignment: .topLeading) {
                TextEditor(text: $m.input)
                    .font(.system(size: 14))
                    .scrollContentBackground(.hidden)
                    .padding(6)
                    .focused($focused)
                    .onKeyPress(.return, phases: .down) { press in
                        if press.modifiers.contains(.shift) { return .ignored }
                        m.send(); return .handled
                    }
                if m.input.isEmpty {
                    Text("Message (Return sends, Shift+Return = new line)").foregroundStyle(.secondary)
                        .padding(.horizontal, 11).padding(.vertical, 6).allowsHitTesting(false)
                }
            }
            .frame(minHeight: 62, maxHeight: 160)
            .fixedSize(horizontal: false, vertical: true)
            .background(RoundedRectangle(cornerRadius: 8).fill(Theme.panel2))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(focused ? Theme.accent : Theme.line))
            HStack(spacing: 12) {
                Button("Send") { m.send() }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(m.busy || m.state == .loading || m.state == .unloading)
                Button("Stop") { m.stop() }
                    .tint(Theme.bad).buttonStyle(.borderedProminent)
                    .keyboardShortcut(".", modifiers: .command)
                    .disabled(!m.busy)
                Button("Clear") { m.clearConversation() }.disabled(m.busy)
                HStack(spacing: 4) {
                    Text("max new tokens").foregroundStyle(.secondary)
                    TextField("", value: $m.maxNew, format: .number).frame(width: 64).textFieldStyle(.roundedBorder)
                }
                Toggle("thinking", isOn: $m.think)
                    .help(m.thinkingLocked != nil ? "fixed for this conversation (Clear to change)" : "Qwen thinking mode")
                    .disabled(m.thinkingLocked != nil)
                if m.think || m.thinkingLocked == true {
                    Picker("effort", selection: $m.reasoningEffort) {
                        Text("xhigh").tag("xhigh"); Text("medium").tag("medium"); Text("low").tag("low")
                    }
                    .frame(width: 130)
                    .help("reasoning_effort of the Qwen template (thinking mode only)")
                }
                Toggle("keep KV", isOn: $m.keepKV)
                    .help("Reuse the kept KV / DeltaNet state for the new turn; off = re-prefill the whole conversation")
                Toggle(mtpLabel, isOn: $m.mtp)
                    .help("MTP speculative decode (draft 1, verify 2): same greedy tokens as plain, fewer passes")
                    .disabled(m.mtpAvailable == false || !m.config.mtp)
                Spacer()
                if !m.status.isEmpty {
                    Text(m.status).font(.caption).foregroundStyle(m.status.contains("STOP") || m.status.contains("missing") ? Theme.bad : Theme.accent)
                        .lineLimit(2).textSelection(.enabled)
                }
            }
            .toggleStyle(.checkbox)
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
        .background(Theme.panel)
        .overlay(alignment: .top) { Rectangle().fill(Theme.line).frame(height: 1) }
        .onAppear { focused = true }
    }

    var mtpLabel: String {
        if !m.config.mtp { return "MTP (started with --no-mtp)" }
        if m.mtpAvailable == false { return "MTP (engine without --mtp)" }
        return "MTP"
    }
}
