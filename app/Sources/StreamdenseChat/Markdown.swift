// Minimal Markdown: fenced code (monospaced, selectable, Copy), headings, lists, tables, rules, paragraphs.
// Inline (bold / italic / code / links) goes through AttributedString(markdown:).
import AppKit
import SwiftUI

enum MDBlock: Hashable {
    case heading(Int, String)
    case paragraph(String)
    case list(ordered: Bool, start: Int, items: [String])
    case code(lang: String, text: String)
    case table(header: [String], rows: [[String]])
    case rule
    case quote(String)
}

enum Markdown {
    static func parse(_ src: String) -> [MDBlock] {
        let lines = src.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n")
        var out: [MDBlock] = []
        var para: [String] = []
        var i = 0
        func flush() { if !para.isEmpty { out.append(.paragraph(para.joined(separator: "\n"))); para = [] } }
        while i < lines.count {
            let l = lines[i]
            let t = l.trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("```") || t.hasPrefix("~~~") {
                flush()
                let fence = String(t.prefix(3))
                let lang = String(t.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                var code: [String] = []
                i += 1
                while i < lines.count && !lines[i].trimmingCharacters(in: .whitespaces).hasPrefix(fence) { code.append(lines[i]); i += 1 }
                i += 1 // closing fence (an unclosed fence while streaming runs to the end)
                out.append(.code(lang: lang, text: code.joined(separator: "\n")))
                continue
            }
            if t.isEmpty { flush(); i += 1; continue }
            if let h = heading(t) { flush(); out.append(.heading(h.0, h.1)); i += 1; continue }
            if t.count >= 3, Set(t.replacingOccurrences(of: " ", with: "")).count == 1, let c = t.first, "-*_".contains(c), para.isEmpty {
                out.append(.rule); i += 1; continue
            }
            if t.contains("|"), i + 1 < lines.count, isTableSep(lines[i + 1]) {
                flush()
                let header = cells(t)
                i += 2
                var rows: [[String]] = []
                while i < lines.count, lines[i].contains("|"), !lines[i].trimmingCharacters(in: .whitespaces).isEmpty {
                    rows.append(cells(lines[i])); i += 1
                }
                out.append(.table(header: header, rows: rows))
                continue
            }
            if let li = listItem(l) {
                flush()
                var items: [String] = [li.text]
                let start = li.number ?? 1
                i += 1
                while i < lines.count {
                    if let n = listItem(lines[i]), n.ordered == li.ordered { items.append(n.text); i += 1; continue }
                    // continuation line of the previous item (indented, not blank)
                    let c = lines[i]
                    if c.hasPrefix("  "), !c.trimmingCharacters(in: .whitespaces).isEmpty, listItem(c) == nil {
                        items[items.count - 1] += " " + c.trimmingCharacters(in: .whitespaces); i += 1; continue
                    }
                    break
                }
                out.append(.list(ordered: li.ordered, start: start, items: items))
                continue
            }
            if t.hasPrefix(">") {
                flush()
                var q: [String] = []
                while i < lines.count, lines[i].trimmingCharacters(in: .whitespaces).hasPrefix(">") {
                    q.append(String(lines[i].trimmingCharacters(in: .whitespaces).dropFirst()).trimmingCharacters(in: .whitespaces)); i += 1
                }
                out.append(.quote(q.joined(separator: "\n")))
                continue
            }
            para.append(l)
            i += 1
        }
        flush()
        return out
    }

    private static func heading(_ t: String) -> (Int, String)? {
        var n = 0
        for c in t { if c == "#" { n += 1 } else { break } }
        guard n >= 1, n <= 6, t.count > n, t[t.index(t.startIndex, offsetBy: n)] == " " else { return nil }
        var s = String(t.dropFirst(n)).trimmingCharacters(in: .whitespaces)
        while s.hasSuffix("#") { s.removeLast() }
        return (n, s.trimmingCharacters(in: .whitespaces))
    }

    private static func isTableSep(_ l: String) -> Bool {
        let t = l.trimmingCharacters(in: .whitespaces)
        guard t.contains("-") else { return false }
        return t.allSatisfy { "|-: ".contains($0) }
    }

    private static func cells(_ l: String) -> [String] {
        var t = l.trimmingCharacters(in: .whitespaces)
        if t.hasPrefix("|") { t.removeFirst() }
        if t.hasSuffix("|") { t.removeLast() }
        return t.components(separatedBy: "|").map { $0.trimmingCharacters(in: .whitespaces) }
    }

    private static func listItem(_ l: String) -> (ordered: Bool, number: Int?, text: String)? {
        let t = l.drop(while: { $0 == " " })
        guard l.count - t.count <= 3 else { return nil }
        if let f = t.first, "-*+".contains(f), t.dropFirst().first == " " {
            return (false, nil, String(t.dropFirst(2)))
        }
        let digits = t.prefix(while: { $0.isNumber })
        if !digits.isEmpty, digits.count <= 9 {
            let rest = t.dropFirst(digits.count)
            if let p = rest.first, p == "." || p == ")", rest.dropFirst().first == " " {
                return (true, Int(digits), String(rest.dropFirst(2)))
            }
        }
        return nil
    }

    static func inline(_ s: String) -> AttributedString {
        let opts = AttributedString.MarkdownParsingOptions(allowsExtendedAttributes: false,
                                                           interpretedSyntax: .inlineOnlyPreservingWhitespace,
                                                           failurePolicy: .returnPartiallyParsedIfPossible)
        var a = (try? AttributedString(markdown: s, options: opts)) ?? AttributedString(s)
        // inline code: tint so it stands out on the dark bubble
        for run in a.runs where run.inlinePresentationIntent?.contains(.code) == true {
            a[run.range].foregroundColor = Color(red: 1.0, green: 0.84, blue: 0.5)
        }
        return a
    }
}

/// Split a reply into (before, thinking, after). Thinking mode: the template's generation prompt may already hold
/// "<think>\n", so a reply can start inside the block (no opening tag).
enum ThinkSplit {
    static func split(_ raw: String, thinkingOn: Bool) -> (thinking: String?, answer: String, thinkingOpen: Bool) {
        var t = raw
        for s in ["<|im_end|>", "<|endoftext|>"] where t.hasSuffix(s) { t.removeLast(s.count) }
        if !t.contains("<think>") && (t.contains("</think>") || (thinkingOn && !t.isEmpty)) { t = "<think>" + t }
        guard let a = t.range(of: "<think>") else { return (nil, t, false) }
        let before = String(t[..<a.lowerBound])
        if let b = t.range(of: "</think>", range: a.upperBound..<t.endIndex) {
            let th = String(t[a.upperBound..<b.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            let rest = String(t[b.upperBound...]).trimmingCharacters(in: .newlines)
            return (th.isEmpty ? nil : th, before + rest, false)
        }
        let th = String(t[a.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        return (th, before, true)
    }
}

// ------------------------------------------------------------------ views
struct MarkdownView: View {
    let text: String
    var body: some View {
        let blocks = Markdown.parse(text)
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, b in
                MDBlockView(block: b)
            }
        }
    }
}

struct MDBlockView: View {
    let block: MDBlock
    var body: some View {
        switch block {
        case .heading(let n, let s):
            Text(Markdown.inline(s))
                .font(.system(size: n == 1 ? 21 : n == 2 ? 18 : 15, weight: .bold))
                .foregroundStyle(.primary)
                .padding(.top, 4)
        case .paragraph(let s):
            Text(Markdown.inline(s)).foregroundStyle(.primary).fixedSize(horizontal: false, vertical: true)
        case .list(let ordered, let start, let items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { k, it in
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text(ordered ? "\(start + k)." : "\u{2022}").foregroundStyle(.primary).monospacedDigit()
                            .frame(minWidth: ordered ? 22 : 12, alignment: .trailing)
                        Text(Markdown.inline(it)).foregroundStyle(.primary).fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        case .code(let lang, let text):
            CodeBlockView(lang: lang, code: text)
        case .table(let header, let rows):
            ScrollView(.horizontal, showsIndicators: true) {
                Grid(alignment: .leading, horizontalSpacing: 0, verticalSpacing: 0) {
                    GridRow {
                        ForEach(Array(header.enumerated()), id: \.offset) { _, h in
                            Text(Markdown.inline(h)).bold().padding(.horizontal, 8).padding(.vertical, 4)
                                .border(Theme.line)
                        }
                    }.background(Theme.panel2)
                    ForEach(Array(rows.enumerated()), id: \.offset) { _, r in
                        GridRow {
                            ForEach(0..<header.count, id: \.self) { k in
                                Text(Markdown.inline(k < r.count ? r[k] : "")).padding(.horizontal, 8).padding(.vertical, 4)
                                    .border(Theme.line)
                            }
                        }
                    }
                }
            }
        case .rule:
            Divider().overlay(Theme.line)
        case .quote(let s):
            HStack(spacing: 8) {
                Rectangle().fill(Theme.accent).frame(width: 3)
                Text(Markdown.inline(s)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

struct CodeBlockView: View {
    let lang: String
    let code: String
    @State private var copied = false
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(lang.isEmpty ? "code" : lang).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                Button(copied ? "Copied" : "Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(code, forType: .string)
                    copied = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { copied = false }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(Color(red: 0.12, green: 0.16, blue: 0.23))
            ScrollView(.horizontal, showsIndicators: true) {
                Text(code)
                    .font(.system(size: 12.5, design: .monospaced))
                    .foregroundStyle(.primary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: true, vertical: true)
                    .padding(10)
            }
        }
        .background(Color(red: 0.02, green: 0.03, blue: 0.05))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(red: 0.23, green: 0.29, blue: 0.39)))
    }
}
