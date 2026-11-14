// System probes in Swift (no subprocesses): RAM / swap / compressor, per-pid CPU + phys_footprint, GPU utilization.
import Darwin
import Foundation
import IOKit

struct VMStats {
    var availGB: Double      // free + inactive + speculative + purgeable (same sum as vm_stat / run_ui.sh)
    var compressorGB: Double // pages occupied by the compressor
    var swapGB: Double       // vm.swapusage used
}

enum SystemStats {
    static let gateAvailGB = 20.0
    static let gateSwapGB = 3.0

    static let pageSize: UInt64 = {
        var ps: vm_size_t = 0
        host_page_size(mach_host_self(), &ps)
        return UInt64(ps)
    }()

    static let tickNs: Double = {
        var tb = mach_timebase_info_data_t()
        mach_timebase_info(&tb)
        return Double(tb.numer) / Double(tb.denom)
    }()

    static func vm() -> VMStats? {
        var info = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        // vm_stat prints "Pages free" = free_count - speculative_count, so free_count already holds speculative.
        let pages = UInt64(info.free_count) + UInt64(info.inactive_count) + UInt64(info.purgeable_count)
        let comp = UInt64(info.compressor_page_count)
        return VMStats(availGB: Double(pages * pageSize) / 1e9,
                       compressorGB: Double(comp * pageSize) / 1e9,
                       swapGB: swapUsedGB())
    }

    static func swapUsedGB() -> Double {
        var x = xsw_usage()
        var sz = MemoryLayout<xsw_usage>.size
        guard sysctlbyname("vm.swapusage", &x, &sz, nil, 0) == 0 else { return 0 }
        return Double(x.xsu_used) / 1e9
    }

    /// (cpu seconds user+sys, phys_footprint bytes) for a pid; nil if it is gone.
    static func pidUsage(_ pid: pid_t) -> (cpuS: Double, footprint: UInt64)? {
        guard pid > 0 else { return nil }
        var ri = rusage_info_v2()
        let rc = withUnsafeMutablePointer(to: &ri) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(pid, RUSAGE_INFO_V2, $0)
            }
        }
        guard rc == 0 else { return nil }
        let cpu = Double(ri.ri_user_time + ri.ri_system_time) * tickNs / 1e9
        return (cpu, ri.ri_phys_footprint)
    }

    /// Direct children of a pid (the watchdog's child is the engine: `nice` execs into it).
    static func childPids(_ pid: pid_t) -> [pid_t] {
        guard pid > 0 else { return [] }
        var buf = [pid_t](repeating: 0, count: 32)
        let n = buf.withUnsafeMutableBufferPointer {
            proc_listchildpids(pid, $0.baseAddress, Int32($0.count * MemoryLayout<pid_t>.size))
        }
        guard n > 0 else { return [] }
        return Array(buf.prefix(Int(n))).filter { $0 > 0 }
    }

    static func isAlive(_ pid: pid_t) -> Bool { pid > 0 && kill(pid, 0) == 0 }

    /// IOAccelerator PerformanceStatistics "Device Utilization %" (same value `ioreg -r -c IOAccelerator` shows).
    static func gpuUtil() -> Int? {
        var it: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &it) == KERN_SUCCESS else { return nil }
        defer { IOObjectRelease(it) }
        var best: Int?
        while case let e = IOIteratorNext(it), e != 0 {
            defer { IOObjectRelease(e) }
            guard let p = IORegistryEntryCreateCFProperty(e, "PerformanceStatistics" as CFString, kCFAllocatorDefault, 0)?
                .takeRetainedValue() as? [String: Any] else { continue }
            if let v = p["Device Utilization %"] as? NSNumber { best = max(best ?? 0, v.intValue) }
        }
        return best
    }

    static func ramGate() -> (ok: Bool, msg: String) {
        guard let s = vm() else { return (false, "cannot read vm statistics") }
        let ok = s.availGB >= gateAvailGB && s.swapGB <= gateSwapGB
        let msg = String(format: "available %.2f GB (need >= %.0f), swap %.2f GB (need <= %.0f)",
                         s.availGB, gateAvailGB, s.swapGB, gateSwapGB)
        return (ok, msg)
    }
}
