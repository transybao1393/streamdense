//! safetensors header index (multi-shard MLX dirs + MTP side file) and a small writer for dumps.
//! Only the 8-byte length + JSON header of each file is read; tensor data is never touched here.
use std::collections::HashMap;
use std::io::{Read, Write};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType { U32, BF16, F16, F32, U8 }

impl DType {
    pub fn size(self) -> usize { match self { DType::U32 | DType::F32 => 4, DType::BF16 | DType::F16 => 2, DType::U8 => 1 } }
    pub fn name(self) -> &'static str {
        match self { DType::U32 => "U32", DType::BF16 => "BF16", DType::F16 => "F16", DType::F32 => "F32", DType::U8 => "U8" }
    }
    pub fn parse(s: &str) -> Option<DType> {
        Some(match s { "U32" => DType::U32, "BF16" => DType::BF16, "F16" => DType::F16, "F32" => DType::F32, "U8" => DType::U8, _ => return None })
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    /// index into `SafeIndex::files()`
    pub file: usize,
    /// absolute file offset of the first data byte (8 + header_len + data_offsets[0])
    pub off: u64,
    pub nbytes: usize,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

pub struct SafeIndex {
    files: Vec<String>,
    names: Vec<String>,
    map: HashMap<String, TensorInfo>,
    /// weight_map entries whose file is missing and holds no `language_model.` tensor (e.g. the vision sidecar)
    pub skipped: Vec<(String, usize)>,
}

impl SafeIndex {
    /// Index an MLX model dir: `model.safetensors.index.json` (weight_map) + every shard header, plus the
    /// file named by `config.json["mtp_file"]` if present. A weight_map file that does not exist is skipped
    /// (listed in `skipped`) only if none of its tensors starts with `language_model.`; otherwise error.
    pub fn open_mlx_dir(dir: &str) -> Result<SafeIndex, String> {
        let dirp = std::path::Path::new(dir);
        let idx_path = dirp.join("model.safetensors.index.json");
        let idx_txt = std::fs::read_to_string(&idx_path).map_err(|e| format!("{}: {}", idx_path.display(), e))?;
        let idx = Json::parse(&idx_txt).map_err(|e| format!("{}: {}", idx_path.display(), e))?;
        let wm = idx.get("weight_map").and_then(|v| v.obj()).ok_or_else(|| format!("{}: no weight_map", idx_path.display()))?;
        // shard file -> names it should hold (ordered by first appearance)
        let mut shards: Vec<(String, Vec<String>)> = Vec::new();
        let mut shard_pos: HashMap<String, usize> = HashMap::new();
        for (name, f) in wm {
            let f = f.str().ok_or_else(|| format!("weight_map[{}] is not a string", name))?;
            let i = *shard_pos.entry(f.to_string()).or_insert_with(|| { shards.push((f.to_string(), Vec::new())); shards.len() - 1 });
            shards[i].1.push(name.clone());
        }
        let mut si = SafeIndex { files: Vec::new(), names: Vec::new(), map: HashMap::new(), skipped: Vec::new() };
        for (rel, want) in &shards {
            let p = dirp.join(rel);
            if !p.exists() {
                if want.iter().any(|n| n.starts_with("language_model.")) {
                    return Err(format!("{} missing but holds language_model tensors", p.display()));
                }
                si.skipped.push((rel.clone(), want.len()));
                continue;
            }
            let fi = si.add_file(&p.to_string_lossy())?;
            for n in want {
                match si.map.get(n) {
                    Some(t) if t.file == fi => {}
                    _ => return Err(format!("{} listed in weight_map for {} but not in its header", n, rel)),
                }
            }
        }
        let cfg_path = dirp.join("config.json");
        if let Ok(cfg_txt) = std::fs::read_to_string(&cfg_path) {
            let cfg = Json::parse(&cfg_txt).map_err(|e| format!("{}: {}", cfg_path.display(), e))?;
            if let Some(mtp) = cfg.get("mtp_file").and_then(|v| v.str()) {
                let p = dirp.join(mtp);
                si.add_file(&p.to_string_lossy())?;
            }
        }
        Ok(si)
    }

    /// Index one safetensors file (header only).
    pub fn open_file(path: &str) -> Result<SafeIndex, String> {
        let mut si = SafeIndex { files: Vec::new(), names: Vec::new(), map: HashMap::new(), skipped: Vec::new() };
        si.add_file(path)?;
        Ok(si)
    }

    fn add_file(&mut self, path: &str) -> Result<usize, String> {
        let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {}", path, e))?;
        let flen = f.metadata().map_err(|e| format!("{}: {}", path, e))?.len();
        let mut lb = [0u8; 8];
        f.read_exact(&mut lb).map_err(|e| format!("{}: header length: {}", path, e))?;
        let hlen = u64::from_le_bytes(lb);
        if hlen > 100 << 20 || 8 + hlen > flen { return Err(format!("{}: bad header length {}", path, hlen)); }
        let mut hb = vec![0u8; hlen as usize];
        f.read_exact(&mut hb).map_err(|e| format!("{}: header: {}", path, e))?;
        let txt = std::str::from_utf8(&hb).map_err(|_| format!("{}: header not utf-8", path))?;
        let h = Json::parse(txt).map_err(|e| format!("{}: {}", path, e))?;
        let obj = h.obj().ok_or_else(|| format!("{}: header is not an object", path))?;
        let fi = self.files.len();
        self.files.push(path.to_string());
        let base = 8 + hlen;
        for (name, v) in obj {
            if name == "__metadata__" { continue; }
            let e = |m: &str| format!("{}: tensor {}: {}", path, name, m);
            let ds = v.get("dtype").and_then(|d| d.str()).ok_or_else(|| e("no dtype"))?;
            let dtype = DType::parse(ds).ok_or_else(|| e(&format!("unsupported dtype {}", ds)))?;
            let shape: Vec<usize> = v.get("shape").and_then(|s| s.arr()).ok_or_else(|| e("no shape"))?
                .iter().map(|x| x.uint().map(|u| u as usize)).collect::<Option<_>>().ok_or_else(|| e("bad shape"))?;
            let offs = v.get("data_offsets").and_then(|s| s.arr()).ok_or_else(|| e("no data_offsets"))?;
            if offs.len() != 2 { return Err(e("data_offsets needs 2 values")); }
            let (a, b) = (offs[0].uint().ok_or_else(|| e("bad offset"))?, offs[1].uint().ok_or_else(|| e("bad offset"))?);
            let nbytes = shape.iter().product::<usize>() * dtype.size();
            if b < a || (b - a) as usize != nbytes { return Err(e(&format!("data_offsets {}..{} != {} bytes for shape", a, b, nbytes))); }
            if base + b > flen { return Err(e("data past end of file")); }
            if self.map.contains_key(name) { return Err(e("duplicate tensor name")); }
            self.names.push(name.clone());
            self.map.insert(name.clone(), TensorInfo { file: fi, off: base + a, nbytes, dtype, shape });
        }
        Ok(fi)
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> { self.map.get(name) }
    /// file paths (dir joined), indexed by `TensorInfo::file`
    pub fn files(&self) -> &[String] { &self.files }
    /// tensor names in file order (shards first, MTP file last)
    pub fn names(&self) -> impl Iterator<Item = &str> { self.names.iter().map(|s| s.as_str()) }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TensorInfo)> { self.names.iter().map(move |n| (n.as_str(), &self.map[n])) }
    pub fn len(&self) -> usize { self.names.len() }
    pub fn is_empty(&self) -> bool { self.names.is_empty() }
}

/// Write a safetensors file: header padded with spaces to 8 bytes, tensors in the given order.
pub fn write_safetensors(path: &str, tensors: &[(&str, DType, &[usize], &[u8])]) -> Result<(), String> {
    let mut hdr = String::from("{\"__metadata__\":{\"format\":\"pt\"}");
    let mut off = 0usize;
    for &(name, dt, shape, data) in tensors {
        let want = shape.iter().product::<usize>() * dt.size();
        if data.len() != want { return Err(format!("{}: {} bytes, shape {:?} {} needs {}", name, data.len(), shape, dt.name(), want)); }
        let sh: Vec<String> = shape.iter().map(|x| x.to_string()).collect();
        hdr.push_str(&format!(",{}:{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}", json_str(name), dt.name(), sh.join(","), off, off + data.len()));
        off += data.len();
    }
    hdr.push('}');
    while hdr.len() % 8 != 0 { hdr.push(' '); }
    let f = std::fs::File::create(path).map_err(|e| format!("{}: {}", path, e))?;
    let mut w = std::io::BufWriter::new(f);
    let e = |x: std::io::Error| format!("{}: {}", path, x);
    w.write_all(&(hdr.len() as u64).to_le_bytes()).map_err(e)?;
    w.write_all(hdr.as_bytes()).map_err(e)?;
    for &(_, _, _, data) in tensors { w.write_all(data).map_err(e)?; }
    w.flush().map_err(e)?;
    Ok(())
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

// ---- minimal JSON reader (no serde in this crate) ----

#[allow(dead_code)]
enum Json { Null, Bool(bool), Num(String), Str(String), Arr(Vec<Json>), Obj(Vec<(String, Json)>) }

impl Json {
    fn parse(s: &str) -> Result<Json, String> {
        let mut p = P { b: s.as_bytes(), i: 0 };
        let v = p.value()?;
        p.ws();
        if p.i != p.b.len() { return Err(format!("json: trailing data at byte {}", p.i)); }
        Ok(v)
    }
    fn get(&self, k: &str) -> Option<&Json> { self.obj()?.iter().find(|(n, _)| n == k).map(|(_, v)| v) }
    fn obj(&self) -> Option<&Vec<(String, Json)>> { if let Json::Obj(o) = self { Some(o) } else { None } }
    fn arr(&self) -> Option<&Vec<Json>> { if let Json::Arr(a) = self { Some(a) } else { None } }
    fn str(&self) -> Option<&str> { if let Json::Str(s) = self { Some(s) } else { None } }
    fn uint(&self) -> Option<u64> { if let Json::Num(n) = self { n.parse().ok() } else { None } }
}

struct P<'a> { b: &'a [u8], i: usize }

impl<'a> P<'a> {
    fn err<T>(&self, m: &str) -> Result<T, String> { Err(format!("json: {} at byte {}", m, self.i)) }
    fn ws(&mut self) { while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') { self.i += 1; } }
    fn lit(&mut self, w: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(w.as_bytes()) { self.i += w.len(); Ok(v) } else { self.err("bad literal") }
    }
    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        if self.i >= self.b.len() { return self.err("unexpected end"); }
        match self.b[self.i] {
            b'{' => {
                self.i += 1;
                let mut o = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') { self.i += 1; return Ok(Json::Obj(o)); }
                loop {
                    self.ws();
                    if self.b.get(self.i) != Some(&b'"') { return self.err("expected key"); }
                    let k = self.string()?;
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') { return self.err("expected ':'"); }
                    self.i += 1;
                    let v = self.value()?;
                    o.push((k, v));
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => { self.i += 1; return Ok(Json::Obj(o)); }
                        _ => return self.err("expected ',' or '}'"),
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') { self.i += 1; return Ok(Json::Arr(a)); }
                loop {
                    a.push(self.value()?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => { self.i += 1; return Ok(Json::Arr(a)); }
                        _ => return self.err("expected ',' or ']'"),
                    }
                }
            }
            b'"' => Ok(Json::Str(self.string()?)),
            b't' => self.lit("true", Json::Bool(true)),
            b'f' => self.lit("false", Json::Bool(false)),
            b'n' => self.lit("null", Json::Null),
            b'-' | b'0'..=b'9' => {
                let st = self.i;
                while self.i < self.b.len() && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') { self.i += 1; }
                Ok(Json::Num(String::from_utf8_lossy(&self.b[st..self.i]).into_owned()))
            }
            _ => self.err("unexpected character"),
        }
    }
    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() { return self.err("short \\u escape"); }
        let h = std::str::from_utf8(&self.b[self.i..self.i + 4]).ok().and_then(|t| u32::from_str_radix(t, 16).ok());
        self.i += 4;
        match h { Some(v) => Ok(v), None => self.err("bad \\u escape") }
    }
    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = match self.b.get(self.i) { Some(&c) => c, None => return self.err("unterminated string") };
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = match self.b.get(self.i) { Some(&e) => e, None => return self.err("bad escape") };
                    self.i += 1;
                    match e {
                        b'"' => out.push(b'"'), b'\\' => out.push(b'\\'), b'/' => out.push(b'/'),
                        b'b' => out.push(8), b'f' => out.push(12), b'n' => out.push(b'\n'), b'r' => out.push(b'\r'), b't' => out.push(b'\t'),
                        b'u' => {
                            let mut cp = self.hex4()?;
                            if (0xd800..0xdc00).contains(&cp) && self.b[self.i..].starts_with(b"\\u") {
                                self.i += 2;
                                let lo = self.hex4()?;
                                cp = 0x10000 + ((cp - 0xd800) << 10) + (lo.wrapping_sub(0xdc00) & 0x3ff);
                            }
                            let ch = char::from_u32(cp).unwrap_or('\u{fffd}');
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return self.err("bad escape"),
                    }
                }
                c => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| format!("json: string not utf-8 at byte {}", self.i))
    }
}
