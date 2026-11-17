//! Minimal HTTP/1.1 on std: request parsing (Content-Length or chunked bodies, keep-alive) and response writing.
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

pub const MAX_BODY: usize = 64 << 20;

#[derive(Debug, Clone)]
pub struct Request {
    pub method: String,
    /// Path without the query string.
    pub path: String,
    pub query: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub http10: bool,
}

impl Request {
    pub fn header(&self, k: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n.eq_ignore_ascii_case(k)).map(|(_, v)| v.as_str())
    }
    pub fn keep_alive(&self) -> bool {
        match self.header("connection").map(|s| s.to_ascii_lowercase()) {
            Some(c) if c.contains("close") => false,
            Some(c) if c.contains("keep-alive") => true,
            _ => !self.http10,
        }
    }
}

/// Error prefix of a socket read that timed out (SO_RCVTIMEO: EAGAIN / os error 35 on macOS, or ETIMEDOUT): the client
/// went quiet, not a bad request. server.rs logs it as `client_timeout` and closes without a 400.
pub const CLIENT_TIMEOUT: &str = "client_timeout: ";
/// Error prefix of any other socket read failure (reset, EOF inside a request): the client went away.
pub const CLIENT_GONE: &str = "client_gone: ";

/// The read timed out before any byte of a line.
const IDLE_TIMEOUT: &str = "client_timeout: idle";

/// A read error as the request reader reports it: timeouts and disconnects are tagged (see `CLIENT_TIMEOUT`).
pub fn io_err(e: std::io::Error) -> String {
    use std::io::ErrorKind::*;
    match e.kind() {
        WouldBlock | TimedOut => format!("{}{}", CLIENT_TIMEOUT, e),
        _ => format!("{}{}", CLIENT_GONE, e),
    }
}

/// The request-reader error is the client's connection (timeout / disconnect), not its request.
pub fn is_client_io(e: &str) -> bool { e.starts_with(CLIENT_TIMEOUT) || e.starts_with(CLIENT_GONE) }

fn read_line<R: BufRead>(r: &mut R, max: usize) -> Result<Option<String>, String> {
    let mut buf = Vec::new();
    let n = loop {
        match r.by_ref().take((max - buf.len()) as u64).read_until(b'\n', &mut buf) {
            Ok(_) => break buf.len(),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            // a timeout before any byte of the line: the caller can tell an idle keep-alive connection apart
            Err(e) if buf.is_empty() && matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => return Err(IDLE_TIMEOUT.into()),
            Err(e) => return Err(io_err(e)),
        }
    };
    if n == 0 { return Ok(None); }
    if !buf.ends_with(b"\n") {
        return Err(if n < max { format!("{}connection closed inside a line", CLIENT_GONE) } else { "header line too long".into() });
    }
    while buf.last() == Some(&b'\n') || buf.last() == Some(&b'\r') { buf.pop(); }
    String::from_utf8(buf).map(Some).map_err(|_| "non-UTF-8 header".into())
}

/// Read one request. Ok(None) = the peer closed the connection cleanly before a request line, or it stayed idle past
/// the read timeout there (a pooled keep-alive connection: closed quietly, never answered with a stale 400).
/// Err: a bad request, or a client timeout / disconnect inside one (`is_client_io`).
pub fn read_request<R: BufRead>(r: &mut R) -> Result<Option<Request>, String> {
    let line = loop {
        let l = match read_line(r, 16 << 10) {
            Err(e) if e == IDLE_TIMEOUT => return Ok(None),
            l => l?,
        };
        match l {
            None => return Ok(None),
            Some(l) if l.is_empty() => continue, // tolerate stray CRLF between requests
            Some(l) => break l,
        }
    };
    let mut it = line.split_whitespace();
    let (method, target, ver) = (it.next().unwrap_or(""), it.next().unwrap_or(""), it.next().unwrap_or(""));
    if method.is_empty() || !target.starts_with('/') || !ver.starts_with("HTTP/1.") { return Err(format!("bad request line {:?}", line)); }
    let (path, query) = match target.split_once('?') { Some((p, q)) => (p.to_string(), q.to_string()), None => (target.to_string(), String::new()) };
    let mut headers = vec![];
    loop {
        let l = read_line(r, 64 << 10)?.ok_or_else(|| format!("{}connection closed inside the headers", CLIENT_GONE))?;
        if l.is_empty() { break; }
        let (k, v) = l.split_once(':').ok_or_else(|| format!("bad header {:?}", l))?;
        headers.push((k.trim().to_string(), v.trim().to_string()));
        if headers.len() > 200 { return Err("too many headers".into()); }
    }
    let mut req = Request { method: method.to_string(), path, query, headers, body: vec![], http10: ver == "HTTP/1.0" };
    let chunked = req.header("transfer-encoding").map(|t| t.to_ascii_lowercase().contains("chunked")).unwrap_or(false);
    if chunked {
        loop {
            let sz = read_line(r, 1024)?.ok_or_else(|| format!("{}connection closed inside a chunk", CLIENT_GONE))?;
            let sz = usize::from_str_radix(sz.split(';').next().unwrap_or("").trim(), 16).map_err(|_| "bad chunk size")?;
            if sz == 0 {
                while let Some(l) = read_line(r, 64 << 10)? { if l.is_empty() { break; } } // trailers
                break;
            }
            // body.len() <= MAX_BODY always holds here, so the subtraction cannot underflow (an addition could wrap)
            if sz > MAX_BODY - req.body.len() { return Err("body too large".into()); }
            let o = req.body.len();
            req.body.resize(o + sz, 0);
            r.read_exact(&mut req.body[o..]).map_err(io_err)?;
            read_line(r, 16)?;
        }
    } else if let Some(cl) = req.header("content-length") {
        let n: usize = cl.parse().map_err(|_| "bad content-length")?;
        if n > MAX_BODY { return Err("body too large".into()); }
        req.body.resize(n, 0);
        r.read_exact(&mut req.body).map_err(io_err)?;
    }
    Ok(Some(req))
}

pub fn reason(code: u16) -> &'static str {
    match code { 200 => "OK", 400 => "Bad Request", 404 => "Not Found", 405 => "Method Not Allowed", 413 => "Payload Too Large",
                 500 => "Internal Server Error", 529 => "Overloaded", _ => "Status" }
}

/// A complete response with Content-Length. `head_only`: headers only (HEAD).
pub fn write_response(w: &mut TcpStream, code: u16, ctype: &str, body: &[u8], keep_alive: bool, head_only: bool, extra: &[(&str, String)]) -> std::io::Result<()> {
    let mut h = format!("HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: {}\r\n",
        code, reason(code), ctype, body.len(), if keep_alive { "keep-alive" } else { "close" });
    for (k, v) in extra { h.push_str(&format!("{}: {}\r\n", k, v)); }
    h.push_str("\r\n");
    w.write_all(h.as_bytes())?;
    if !head_only { w.write_all(body)?; }
    w.flush()
}

/// Start an SSE response. The stream is not length-delimited: the connection closes after the last event.
pub fn write_sse_head(w: &mut TcpStream, extra: &[(&str, String)]) -> std::io::Result<()> {
    let mut h = String::from("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\n");
    for (k, v) in extra { h.push_str(&format!("{}: {}\r\n", k, v)); }
    h.push_str("\r\n");
    w.write_all(h.as_bytes())?;
    w.flush()
}

pub fn reader(s: &TcpStream) -> std::io::Result<BufReader<TcpStream>> { Ok(BufReader::new(s.try_clone()?)) }

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parse_requests() {
        let raw = b"POST /v1/messages?beta=true HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\nhello\
GET /v1/models HTTP/1.1\r\nConnection: close\r\n\r\n\
POST /c HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2;x=1\r\nde\r\n0\r\n\r\n";
        let mut r = Cursor::new(&raw[..]);
        let a = read_request(&mut r).unwrap().unwrap();
        assert_eq!((a.method.as_str(), a.path.as_str(), a.query.as_str(), &a.body[..]), ("POST", "/v1/messages", "beta=true", &b"hello"[..]));
        assert!(a.keep_alive());
        assert_eq!(a.header("HOST"), Some("x"));
        let b = read_request(&mut r).unwrap().unwrap();
        assert_eq!(b.path, "/v1/models");
        assert!(!b.keep_alive());
        let c = read_request(&mut r).unwrap().unwrap();
        assert_eq!(c.body, b"abcde");
        assert!(read_request(&mut r).unwrap().is_none());
        assert!(read_request(&mut Cursor::new(&b"garbage\r\n\r\n"[..])).is_err());
    }

    /// Yields its bytes, then fails every read with `kind` (EAGAIN = SO_RCVTIMEO expiry on macOS).
    struct Stall { data: Vec<u8>, at: usize, kind: std::io::ErrorKind, intr: bool }
    impl Read for Stall {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            if self.intr { self.intr = false; return Err(std::io::ErrorKind::Interrupted.into()); }
            if self.at >= self.data.len() { return Err(std::io::Error::from(self.kind)); }
            let n = b.len().min(self.data.len() - self.at).min(7);
            b[..n].copy_from_slice(&self.data[self.at..self.at + n]);
            self.at += n;
            Ok(n)
        }
    }

    #[test]
    fn timeouts_are_not_bad_requests() {
        use std::io::ErrorKind::*;
        let rd = |s: &[u8], kind| BufReader::new(Stall { data: s.to_vec(), at: 0, kind, intr: true });
        // idle keep-alive connection timing out before a request: a quiet close
        for k in [WouldBlock, TimedOut] { assert!(read_request(&mut rd(b"", k)).unwrap().is_none()); }
        // a complete request is read through EINTR, the timeout after it ends the next read quietly
        let mut r = rd(b"GET /health/ram HTTP/1.1\r\n\r\n", WouldBlock);
        assert_eq!(read_request(&mut r).unwrap().unwrap().path, "/health/ram");
        assert!(read_request(&mut r).unwrap().is_none());
        // a timeout / reset inside a request: a client error, never "bad request"
        for (s, k) in [(&b"GET /x HT"[..], WouldBlock), (b"GET /x HTTP/1.1\r\nHost: a\r\n", WouldBlock),
                       (b"POST /v HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc", TimedOut), (b"POST /v HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc", ConnectionReset)] {
            let e = read_request(&mut rd(s, k)).unwrap_err();
            assert!(is_client_io(&e), "{:?}: {}", String::from_utf8_lossy(s), e);
            assert_eq!(e.starts_with(CLIENT_TIMEOUT), k != ConnectionReset, "{}", e);
        }
        assert!(is_client_io(&read_request(&mut Cursor::new(&b"GET / HTTP/1.1\r\nHost"[..])).unwrap_err()), "EOF inside the headers");
        assert!(!is_client_io(&read_request(&mut Cursor::new(&b"garbage\r\n\r\n"[..])).unwrap_err()));
    }

    #[test]
    fn huge_chunk_size_is_rejected_not_wrapped() {
        // usize::MAX as a chunk size: `len + sz` would wrap in release builds; it must be "body too large", no panic
        let raw = b"POST /c HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\nffffffffffffffff\r\nx\r\n0\r\n\r\n";
        assert_eq!(read_request(&mut Cursor::new(&raw[..])).unwrap_err(), "body too large");
        let raw = b"POST /c HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n4000001\r\n";
        assert_eq!(read_request(&mut Cursor::new(&raw[..])).unwrap_err(), "body too large");
    }
}
