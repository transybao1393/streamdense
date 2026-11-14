//! Minimal blocking HTTP client for tests and smoke checks (std only): one request per connection.
use serde_json::Value;
use std::io::{BufRead, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Response { pub status: u16, pub headers: Vec<(String, String)>, pub body: Vec<u8> }

impl Response {
    pub fn header(&self, k: &str) -> Option<&str> { self.headers.iter().find(|(n, _)| n.eq_ignore_ascii_case(k)).map(|(_, v)| v.as_str()) }
    pub fn json(&self) -> Value { serde_json::from_slice(&self.body).unwrap_or_else(|e| panic!("non-JSON body ({}): {}", e, String::from_utf8_lossy(&self.body))) }
    pub fn text(&self) -> String { String::from_utf8_lossy(&self.body).to_string() }
    /// SSE events as (event name, data JSON), in order.
    pub fn events(&self) -> Vec<(String, Value)> { parse_sse(&self.text()) }
}

pub fn parse_sse(s: &str) -> Vec<(String, Value)> {
    let mut out = vec![];
    for blk in s.split("\n\n") {
        let (mut name, mut data) = (String::new(), String::new());
        for l in blk.lines() {
            if let Some(x) = l.strip_prefix("event: ") { name = x.to_string(); }
            else if let Some(x) = l.strip_prefix("data: ") { data.push_str(x); }
        }
        if !name.is_empty() || !data.is_empty() {
            out.push((name, serde_json::from_str(&data).unwrap_or(Value::String(data))));
        }
    }
    out
}

/// Send one request (Connection: close) and read the whole response (Content-Length or until close).
pub fn request(port: u16, method: &str, path: &str, body: Option<&Value>) -> std::io::Result<Response> {
    request_with(port, method, path, body, &[])
}

/// `request` with extra headers (e.g. `x-streamdense-warm: 1`).
pub fn request_with(port: u16, method: &str, path: &str, body: Option<&Value>, headers: &[(&str, &str)]) -> std::io::Result<Response> {
    let mut s = TcpStream::connect(("127.0.0.1", port))?;
    s.set_read_timeout(Some(Duration::from_secs(300)))?;
    let b = body.map(|v| v.to_string()).unwrap_or_default();
    let extra: String = headers.iter().map(|(k, v)| format!("{}: {}\r\n", k, v)).collect();
    let req = format!("{} {} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nContent-Type: application/json\r\nanthropic-version: 2023-06-01\r\n{}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        method, path, port, extra, b.len(), b);
    s.write_all(req.as_bytes())?;
    let mut r = std::io::BufReader::new(s);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let status: u16 = line.split_whitespace().nth(1).and_then(|x| x.parse().ok())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad status line {:?}", line)))?;
    let mut headers = vec![];
    loop {
        let mut l = String::new();
        r.read_line(&mut l)?;
        let l = l.trim_end();
        if l.is_empty() { break; }
        if let Some((k, v)) = l.split_once(':') { headers.push((k.trim().to_string(), v.trim().to_string())); }
    }
    let mut body = vec![];
    let cl = headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("content-length")).and_then(|(_, v)| v.parse::<usize>().ok());
    if method == "HEAD" {
    } else if let Some(n) = cl { body.resize(n, 0); r.read_exact(&mut body)?; } else { r.read_to_end(&mut body)?; }
    Ok(Response { status, headers, body })
}
