//! Local calendar date for chat templates. transformers defines `strftime_now` with the LOCAL time
//! (`datetime.now()`), so the "Today Date" lines of the SmolLM3 and Llama 3.1 / 3.2 templates use the local date.
//! The local UTC offset comes from libc `localtime_r` (declared here: no libc crate dependency).

const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                            "October", "November", "December"];

/// Seconds since the Unix epoch (0 when the clock is before it).
pub fn now_secs() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// libc `struct tm` (macOS / glibc layout: nine ints, then `tm_gmtoff`, `tm_zone`).
#[repr(C)]
struct Tm {
    tm_sec: i32, tm_min: i32, tm_hour: i32, tm_mday: i32, tm_mon: i32, tm_year: i32, tm_wday: i32, tm_yday: i32,
    tm_isdst: i32, tm_gmtoff: std::os::raw::c_long, tm_zone: *const std::os::raw::c_char,
}

extern "C" {
    fn localtime_r(t: *const i64, out: *mut Tm) -> *mut Tm;
}

/// Local UTC offset in seconds at Unix time `secs` (`localtime_r`'s `tm_gmtoff`); `None` when libc fails.
pub fn local_offset_at(secs: i64) -> Option<i64> {
    let mut tm: Tm = Tm { tm_sec: 0, tm_min: 0, tm_hour: 0, tm_mday: 0, tm_mon: 0, tm_year: 0, tm_wday: 0, tm_yday: 0,
                          tm_isdst: 0, tm_gmtoff: 0, tm_zone: std::ptr::null() };
    // SAFETY: both pointers are valid for the call; localtime_r is the reentrant form and writes only `tm`.
    let r = unsafe { localtime_r(&secs, &mut tm) };
    if r.is_null() { None } else { Some(tm.tm_gmtoff as i64) }
}

/// Local day number (days since 1970-01-01) at Unix time `secs` (UTC when the local offset is unknown).
pub fn local_days_at(secs: i64) -> i64 {
    (secs + local_offset_at(secs).unwrap_or(0)).div_euclid(86400)
}

/// Days since 1970-01-01 -> (year, month 1..=12, day 1..=31) (civil_from_days, H. Hinnant).
pub fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    (year, month, day)
}

/// Days since 1970-01-01 -> `%d %B %Y` ("15 November 2026").
pub fn date_from_days(days: i64) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{:02} {} {}", d, MONTHS[(m - 1) as usize], y)
}

/// Days since 1970-01-01 -> `%d %b %Y` ("15 Nov 2026").
pub fn date_from_days_short(days: i64) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{:02} {} {}", d, &MONTHS[(m - 1) as usize][..3], y)
}

/// Local date at Unix time `secs` as `%d %B %Y` (UTC when the local offset is unknown).
pub fn local_date_at(secs: i64) -> String { date_from_days(local_days_at(secs)) }

/// Local date at Unix time `secs` as `%d %b %Y` (UTC when the local offset is unknown).
pub fn local_date_short_at(secs: i64) -> String { date_from_days_short(local_days_at(secs)) }

/// Today's local date as `%d %B %Y`, what transformers' `strftime_now("%d %B %Y")` writes.
pub fn today_local() -> String { local_date_at(now_secs()) }

/// Today's local date as `%d %b %Y`, what transformers' `strftime_now("%d %b %Y")` writes.
pub fn today_local_short() -> String { local_date_short_at(now_secs()) }

/// Today's UTC date as `%d %B %Y`.
pub fn today_utc() -> String { date_from_days(now_secs().div_euclid(86400)) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(date_from_days(0), "01 January 1970");
        assert_eq!(date_from_days_short(0), "01 Jan 1970");
        assert_eq!(date_from_days_short(20407), "15 Nov 2025");
        assert_eq!(date_from_days_short(20772), "15 Nov 2026");
        assert_eq!(date_from_days_short(20513), "01 Mar 2026");
        assert_eq!(date_from_days_short(19782), "29 Feb 2024");
    }

    #[test]
    fn short_follows_local_offset() {
        // 2026-11-15 23:30 UTC and 2026-11-16 00:30 UTC: the date follows the local offset across midnight.
        for t in [1_794_785_400i64, 1_794_789_000, 0, 1_700_000_000] {
            let off = local_offset_at(t).expect("localtime_r");
            assert_eq!(local_date_short_at(t), date_from_days_short((t + off).div_euclid(86400)), "t={}", t);
            assert_eq!(local_date_at(t), date_from_days((t + off).div_euclid(86400)), "t={}", t);
        }
    }
}
