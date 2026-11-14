// Live side panel: speed of the last reply, RAM vs cap, CPU / GPU, sparklines, reply history, watchdog log.
import SwiftUI

struct Card<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased()).font(.system(size: 12, weight: .semibold)).kerning(0.5).foregroundStyle(.primary)
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12).padding(.vertical, 11)
        .background(RoundedRectangle(cornerRadius: 10).fill(Theme.panel))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Theme.line))
    }
}

struct Big: View {
    let value: String
    let label: String
    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(value).font(.system(size: 24, weight: .bold)).monospacedDigit().foregroundStyle(.primary)
            Text(label).font(.system(size: 11)).foregroundStyle(.secondary)
        }
    }
}

struct KV: View {
    let rows: [(String, String)]
    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 3) {
            ForEach(Array(rows.enumerated()), id: \.offset) { _, r in
                GridRow {
                    Text(r.0).foregroundStyle(.secondary)
                    Text(r.1).foregroundStyle(.primary).fontWeight(.semibold).monospacedDigit()
                        .gridColumnAlignment(.trailing).multilineTextAlignment(.trailing)
                        .textSelection(.enabled)
                }
            }
        }
        .font(.system(size: 12))
    }
}

struct Sparkline: View {
    let values: [Double?]
    var cap: Double? = nil
    var body: some View {
        Canvas { ctx, size in
            let v = values.compactMap { $0 }
            guard !v.isEmpty else {
                ctx.draw(Text("no data yet").font(.system(size: 11)).foregroundColor(.secondary), at: CGPoint(x: 40, y: size.height / 2))
                return
            }
            let hi = max(v.max() ?? 0, cap ?? 0) * 1.08
            let w = size.width, h = size.height
            func y(_ x: Double) -> CGFloat { h - CGFloat(x / max(hi, 1e-9)) * (h - 6) - 3 }
            if let cap {
                var p = Path(); p.move(to: CGPoint(x: 0, y: y(cap))); p.addLine(to: CGPoint(x: w, y: y(cap)))
                ctx.stroke(p, with: .color(Theme.bad), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
            }
            var p = Path()
            var started = false
            let n = values.count
            for (i, x) in values.enumerated() {
                guard let x else { started = false; continue }
                let X = n < 2 ? w / 2 : CGFloat(i) / CGFloat(n - 1) * (w - 6) + 3
                let pt = CGPoint(x: X, y: y(x))
                if started { p.addLine(to: pt) } else { p.move(to: pt); started = true }
                if n < 40 { ctx.fill(Path(CGRect(x: X - 1.5, y: pt.y - 1.5, width: 3, height: 3)), with: .color(Theme.accent)) }
            }
            ctx.stroke(p, with: .color(Theme.accent), lineWidth: 1.6)
            ctx.draw(Text(fmt(v.last, 1)).font(.system(size: 11, weight: .semibold)).foregroundColor(.white),
                     at: CGPoint(x: w - 20, y: 9))
        }
        .frame(height: 54)
        .background(RoundedRectangle(cornerRadius: 5).fill(Theme.panel2))
    }
}

struct FootprintBar: View {
    let fp: Double?
    let peak: Double?
    let cap: Double
    var body: some View {
        GeometryReader { g in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 5).fill(Theme.panel2)
                if let fp {
                    Rectangle().fill(fp / cap > 0.95 ? Theme.bad : Theme.accent)
                        .frame(width: g.size.width * min(1, fp / cap))
                }
                if let peak, peak > 0 {
                    Rectangle().fill(Theme.warn).frame(width: 2).offset(x: g.size.width * min(0.995, peak / cap))
                }
                Text(fp.map { "\(fmt($0, 2)) GB of \(fmt(cap, 1)) GB cap (\(fmt($0 / cap * 100, 0))%) · peak \(fmt(peak, 2))" }
                     ?? "engine not loaded · cap \(fmt(cap, 1)) GB")
                    .font(.system(size: 11, weight: .semibold)).foregroundStyle(.primary)
                    .frame(maxWidth: .infinity)
            }
            .clipShape(RoundedRectangle(cornerRadius: 5))
            .overlay(RoundedRectangle(cornerRadius: 5).stroke(Theme.line))
        }
        .frame(height: 18)
    }
}

struct SidePanel: View {
    @EnvironmentObject var m: ChatModel
    var body: some View {
        ScrollView {
            VStack(spacing: 12) {
                speedCard
                ramCard
                Card(title: "CPU / GPU") {
                    HStack(alignment: .top, spacing: 18) {
                        Big(value: fmt(m.sample.engineCpu, 0), label: "engine CPU %")
                        Big(value: m.sample.gpu.map(String.init) ?? "-", label: "GPU util %")
                        Big(value: fmt(m.sample.appCpu, 0), label: "app CPU %")
                    }
                }
                Card(title: "Trends") {
                    Text("decode tok/s per reply").font(.system(size: 12)).foregroundStyle(.secondary)
                    Sparkline(values: m.replies.map { $0.done.decodeTokS })
                    Text("engine RAM GB (1 Hz, last 10 min) · dashed = \(fmt(EngineConfig.capGB, 1)) GB cap").font(.system(size: 12)).foregroundStyle(.secondary)
                    Sparkline(values: m.ramHist, cap: EngineConfig.capGB)
                }
                historyCard
                watchdogCard
            }
            .padding(12)
        }
        .background(Theme.bg)
    }

    var speedCard: some View {
        let d = m.lastReply?.done
        return Card(title: "Speed (last reply)") {
            HStack(alignment: .top, spacing: 16) {
                Big(value: fmt(d?.decodeTokS, 2), label: "decode tok/s")
                Big(value: fmt(d?.prefillTokS, 1), label: "prefill tok/s")
                Big(value: fmt(d?.ttftMs, 0), label: "TTFT ms")
                Big(value: d == nil ? "-" : d!.spec.map { fmt($0.acceptance * 100, 0) } ?? "off", label: "MTP accept %")
            }
            if let r = m.lastReply {
                KV(rows: speedRows(r))
            } else {
                Text("no reply yet").foregroundStyle(.secondary).font(.system(size: 12))
            }
        }
    }

    func speedRows(_ r: ReplyRow) -> [(String, String)] {
        let d = r.done
        var rows: [(String, String)] = [
            ("prompt tokens", "\(d.nPrompt.map(String.init) ?? "-")" + (d.kvReused ? " (KV reused)" : " (full prefill)")),
            ("KV reused / fed", "\(d.kvReusedTokens.map(String.init) ?? "-") / \(d.nFeed.map(String.init) ?? "-")" + (d.kvMode.map { " (\($0))" } ?? "")),
            ("prefill time", "\(fmt(d.prefillMs, 0)) ms"),
            ("TTFT engine / app", "\(fmt(d.ttftMs, 0)) / \(fmt(r.uiTtftMs, 0)) ms"),
            ("generated tokens", "\(d.nNew)"),
            ("ms / token (wall, GPU)", "\(fmt(d.msPerTok, 2)), \(fmt(d.gpuMsPerTok, 2))"),
            ("decode mode", d.mtpUsed ? "MTP (draft 1, verify 2)" : (d.mtpRequested ? "plain (engine has no MTP)" : "plain")),
        ]
        if let sp = d.spec {
            rows.append(("MTP acceptance", "\(fmt(sp.acceptance * 100, 1))%" + (sp.passes.map { p in " (\(sp.accepted ?? 0) of \(p) passes)" } ?? "")))
            rows.append(("MTP tokens per pass", fmt(sp.tokensPerPass, 2)))
            rows.append(("MTP speedup vs plain", r.mtpSpeedup.map { "\(fmt($0, 2))x (vs #\(r.mtpSpeedupRef ?? 0))" } ?? "- (no plain reply yet)"))
        }
        rows.append(("context used / max", "\(d.ctxUsed.map(String.init) ?? "-") / \(d.maxCtx ?? m.config.maxCtx)"))
        rows.append(("stop reason", d.stopReason))
        rows.append(("engine footprint at end", "\(fmt(d.footprintGB, 3)) GB"))
        rows.append(("engine peak (engine monitor)", "\(fmt(d.peakFootprintGB, 3)) GB"))
        return rows
    }

    var ramCard: some View {
        let s = m.sample
        let peak = max(m.peakFpGB, m.lastReply?.done.peakFootprintGB ?? 0)
        return Card(title: "RAM") {
            HStack(alignment: .top, spacing: 18) {
                Big(value: fmt(s.engineFpGB, 2), label: "engine phys_footprint GB")
                Big(value: fmt(s.vm?.availGB, 1), label: "system available GB")
            }
            FootprintBar(fp: s.engineFpGB, peak: peak > 0 ? peak : nil, cap: EngineConfig.capGB)
            KV(rows: [
                ("engine current / peak", "\(fmt(s.engineFpGB, 3)) / \(fmt(peak > 0 ? peak : nil, 3)) GB"),
                ("system available", "\(fmt(s.vm?.availGB, 2)) GB"),
                ("swap used", "\(fmt(s.vm?.swapGB, 2)) GB"),
                ("compressor", "\(fmt(s.vm?.compressorGB, 2)) GB"),
                ("app footprint", "\(fmt(s.appFpGB, 3)) GB"),
                ("load gate", "available >= \(Int(SystemStats.gateAvailGB)) GB, swap <= \(Int(SystemStats.gateSwapGB)) GB"),
            ])
        }
    }

    var historyCard: some View {
        Card(title: "Replies this session") {
            ScrollView(.horizontal) {
                Grid(alignment: .trailing, horizontalSpacing: 8, verticalSpacing: 3) {
                    GridRow {
                        ForEach(["#", "prompt", "gen", "prefill t/s", "decode t/s", "ms/tok", "MTP", "peak GB", "stop"], id: \.self) {
                            Text($0).fontWeight(.semibold).foregroundStyle(.primary)
                        }
                    }
                    Divider().overlay(Theme.line).gridCellUnsizedAxes(.horizontal)
                    ForEach(m.replies.reversed()) { r in
                        let d = r.done
                        GridRow {
                            Text("\(r.n)\(d.kvReused ? "c" : "")").foregroundStyle(.primary).gridColumnAlignment(.leading)
                            Text(d.nPrompt.map(String.init) ?? "-")
                            Text("\(d.nNew)")
                            Text(fmt(d.prefillTokS, 1))
                            Text(fmt(d.decodeTokS, 2))
                            Text(fmt(d.msPerTok, 1))
                            Text(mtpCell(r)).foregroundStyle(.primary)
                            Text(fmt(max(d.peakFootprintGB ?? 0, r.enginePeakGB ?? 0), 2))
                            Text(d.stopReason)
                        }
                        .foregroundStyle(.secondary)
                    }
                }
                .font(.system(size: 12)).monospacedDigit()
            }
            if m.replies.isEmpty { Text("none yet").font(.system(size: 12)).foregroundStyle(.secondary) }
        }
    }

    func mtpCell(_ r: ReplyRow) -> String {
        guard let sp = r.done.spec else { return "off" }
        var s = "\(fmt(sp.acceptance * 100, 0))%/\(fmt(sp.tokensPerPass, 2))"
        if let x = r.mtpSpeedup { s += "/\(fmt(x, 2))x" }
        return s
    }

    var watchdogCard: some View {
        let wd = m.engine.isRunning
            ? "running (pid \(m.engine.watchdogPid)): cap \(fmt(EngineConfig.capGB, 1)) GB + margin 0, swap limit \(Int(EngineConfig.swapGB)) GB"
            : "not running"
        let info = m.readyInfo
        return Card(title: "Watchdog / engine") {
            KV(rows: [
                ("watchdog", wd),
                ("last watchdog line", m.watchdogLast.isEmpty ? "-" : m.watchdogLast),
                ("engine pid", m.engine.currentEnginePid() > 0 ? "\(m.engine.currentEnginePid())" : "-"),
                ("load time", m.loadSeconds.map { "\(fmt($0, 1)) s" } ?? "-"),
                ("decode / prefill", (info["decode"] as? String).map { "\($0), prefill K=\(info["prefill_k"] ?? "-")" } ?? "-"),
                ("chat (tokenizer)", m.chatAvailable.map { $0 ? "available" : "NOT available (see log)" } ?? "-"),
                ("MTP", m.mtpAvailable.map { $0 ? "loaded (--mtp)" : "not loaded" } ?? (m.config.mtp ? "on (--mtp at next load)" : "off (--no-mtp)")),
                ("conversation", "\(m.turns) turns" + (m.thinkingLocked.map { $0 ? " · thinking on" : " · thinking off" } ?? "") + (m.kvKept ? " · KV kept" : "")),
            ])
            ScrollViewReader { proxy in
                ScrollView {
                    Text(m.log.suffix(60).joined(separator: "\n"))
                        .font(.system(size: 11, design: .monospaced)).foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                    Color.clear.frame(height: 1).id("logEnd")
                }
                .frame(height: 140)
                .onChange(of: m.log.count) { _, _ in proxy.scrollTo("logEnd", anchor: .bottom) }
            }
        }
    }
}
