//! Hugging Face helpers: resolve URLs, list repo files, read a remote
//! safetensors header, and fetch a subset of tensors into a valid
//! safetensors file using ranged GETs.

use std::fs::OpenOptions;
use std::os::unix::fs::FileExt;
use std::path::Path;

use serde_json::{Map, Value};

use super::{FetchConfig, Range, Stats};

const HF: &str = "https://huggingface.co";
/// Refuse absurd header lengths (corrupt file or wrong URL).
const MAX_HEADER: u64 = 256 << 20;

/// Percent-encode '/' in a revision (e.g. `refs/pr/1`).
fn enc_rev(revision: &str) -> String {
    revision.replace('/', "%2F")
}

/// `https://huggingface.co/{repo}/resolve/{rev}/{file}`
pub fn resolve_url(repo: &str, revision: &str, file: &str) -> String {
    format!("{HF}/{repo}/resolve/{}/{file}", enc_rev(revision))
}

#[derive(Debug, Clone)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
}

/// Next-page URL from a `Link: <url>; rel="next"` header.
fn next_link(link: &str) -> Option<String> {
    for part in link.split(',') {
        let part = part.trim();
        if part.contains("rel=\"next\"") || part.contains("rel=next") {
            let s = part.find('<')?;
            let e = part[s + 1..].find('>')? + s + 1;
            let u = &part[s + 1..e];
            return Some(if u.starts_with('/') { format!("{HF}{u}") } else { u.to_string() });
        }
    }
    None
}

/// List every file in the repo at `revision` (recursive, paginated).
pub fn list_files(repo: &str, revision: &str, cfg: &FetchConfig) -> Result<Vec<RepoFile>, String> {
    let agent = super::agent(cfg);
    let mut url = Some(format!("{HF}/api/models/{repo}/tree/{}?recursive=true", enc_rev(revision)));
    let mut out = Vec::new();
    let mut pages = 0usize;
    while let Some(u) = url.take() {
        pages += 1;
        if pages > 10_000 {
            return Err("list_files: too many pages".into());
        }
        let mut req = agent.get(&u);
        if let Some(h) = super::auth_header(&u, cfg) {
            req = req.header("Authorization", h);
        }
        let mut resp = req.call().map_err(|e| format!("GET {u}: {e}"))?;
        let status = resp.status().as_u16();
        let link = super::header_str(&resp, "link");
        let body = resp
            .body_mut()
            .with_config()
            .limit(256 << 20)
            .read_to_string()
            .map_err(|e| format!("GET {u}: read body: {e}"))?;
        if status != 200 {
            let snip: String = body.chars().take(300).collect();
            return Err(format!("GET {u}: HTTP {status}: {snip}"));
        }
        let v: Value = serde_json::from_str(&body).map_err(|e| format!("tree JSON: {e}"))?;
        let arr = v.as_array().ok_or("tree JSON: expected an array")?;
        for it in arr {
            if it.get("type").and_then(Value::as_str) != Some("file") {
                continue;
            }
            let Some(path) = it.get("path").and_then(Value::as_str) else { continue };
            let size = it
                .get("lfs")
                .and_then(|l| l.get("size"))
                .and_then(Value::as_u64)
                .or_else(|| it.get("size").and_then(Value::as_u64))
                .unwrap_or(0);
            out.push(RepoFile { path: path.to_string(), size });
        }
        url = link.as_deref().and_then(next_link);
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Absolute byte offsets in the file (start inclusive, end exclusive).
    pub start: u64,
    pub end: u64,
}

/// Read the safetensors header of a remote file.
/// Returns `(data_start, tensors sorted by name, raw header JSON)`.
pub fn safetensors_header(url: &str, cfg: &FetchConfig) -> Result<(u64, Vec<TensorInfo>, Value), String> {
    let b = super::get_range(url, 0, 8, cfg)?;
    if b.len() != 8 {
        return Err(format!("safetensors: header length read {} bytes, want 8", b.len()));
    }
    let n = u64::from_le_bytes(b[..8].try_into().unwrap());
    if n == 0 || n > MAX_HEADER {
        return Err(format!("safetensors: implausible header length {n}"));
    }
    let hb = super::get_range(url, 8, 8 + n, cfg)?;
    if hb.len() as u64 != n {
        return Err(format!("safetensors: header read {} bytes, want {n}", hb.len()));
    }
    let v: Value = serde_json::from_slice(&hb).map_err(|e| format!("safetensors header JSON: {e}"))?;
    let obj = v.as_object().ok_or("safetensors header: not an object")?;
    let data_start = 8 + n;
    let mut ts = Vec::with_capacity(obj.len());
    for (name, t) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = t
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("tensor {name}: missing dtype"))?
            .to_string();
        let shape = t
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("tensor {name}: missing shape"))?
            .iter()
            .map(|d| d.as_u64().ok_or_else(|| format!("tensor {name}: bad shape")))
            .collect::<Result<Vec<u64>, String>>()?;
        let off = t
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|a| a.len() == 2)
            .ok_or_else(|| format!("tensor {name}: missing data_offsets"))?;
        let (s, e) = match (off[0].as_u64(), off[1].as_u64()) {
            (Some(s), Some(e)) if e >= s => (s, e),
            _ => return Err(format!("tensor {name}: bad data_offsets")),
        };
        ts.push(TensorInfo { name: name.clone(), dtype, shape, start: data_start + s, end: data_start + e });
    }
    ts.sort_by(|a, b| a.name.cmp(&b.name));
    Ok((data_start, ts, v))
}

/// Fetch the tensors of `repo/file` whose names pass `filter` into a valid
/// safetensors file at `out_path` (original names, dtypes, shapes; data packed
/// in name order). Returns fetch stats and the tensor count.
pub fn fetch_tensors(
    repo: &str,
    revision: &str,
    file: &str,
    filter: &dyn Fn(&str) -> bool,
    out_path: &Path,
    cfg: &FetchConfig,
) -> Result<(Stats, usize), String> {
    let h = super::head(&resolve_url(repo, revision, file), cfg)?;
    let url = h.final_url.as_str();
    let (_, all, raw) = safetensors_header(url, cfg)?;
    let pick: Vec<&TensorInfo> = all.iter().filter(|t| filter(&t.name)).collect();
    if pick.is_empty() {
        return Err(format!("no tensors in {file} match the filter ({} total)", all.len()));
    }
    for t in &pick {
        if h.size > 0 && t.end > h.size {
            return Err(format!("tensor {}: end {} beyond file size {}", t.name, t.end, h.size));
        }
    }

    // New header: same entries, packed data_offsets in name order.
    let mut hdr = Map::new();
    if let Some(m) = raw.get("__metadata__") {
        hdr.insert("__metadata__".into(), m.clone());
    }
    let mut off = 0u64;
    let mut rel = Vec::with_capacity(pick.len());
    for t in &pick {
        let len = t.end - t.start;
        let mut e = Map::new();
        e.insert("dtype".into(), Value::String(t.dtype.clone()));
        e.insert("shape".into(), Value::Array(t.shape.iter().map(|&d| Value::from(d)).collect()));
        e.insert("data_offsets".into(), Value::Array(vec![Value::from(off), Value::from(off + len)]));
        hdr.insert(t.name.clone(), Value::Object(e));
        rel.push(off);
        off += len;
    }
    let data_len = off;
    let mut hjson = serde_json::to_vec(&Value::Object(hdr)).map_err(|e| format!("header JSON: {e}"))?;
    while hjson.len() % 8 != 0 {
        hjson.push(b' ');
    }
    let hlen = hjson.len() as u64;
    let data_base = 8 + hlen;

    let out = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(out_path)
        .map_err(|e| format!("open {}: {e}", out_path.display()))?;
    out.set_len(data_base + data_len).map_err(|e| format!("set_len {}: {e}", out_path.display()))?;
    out.write_all_at(&hlen.to_le_bytes(), 0).map_err(|e| format!("write header: {e}"))?;
    out.write_all_at(&hjson, 8).map_err(|e| format!("write header: {e}"))?;

    let ranges: Vec<Range> = pick
        .iter()
        .zip(&rel)
        .filter(|(t, _)| t.end > t.start)
        .map(|(t, &r)| Range { src_offset: t.start, len: t.end - t.start, dst_offset: data_base + r })
        .collect();
    let stats = super::fetch_ranges(url, &ranges, &out, cfg, None)?;
    out.sync_all().map_err(|e| format!("sync {}: {e}", out_path.display()))?;
    Ok((stats, pick.len()))
}

/// Glob match where only `*` is special (matches any run, including '/').
pub fn glob_match(pattern: &str, s: &str) -> bool {
    let p = pattern.as_bytes();
    let t = s.as_bytes();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut mark) = (None::<usize>, 0usize);
    while ti < t.len() {
        if pi < p.len() && p[pi] == b'*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if pi < p.len() && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if let Some(sp) = star {
            pi = sp + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}
