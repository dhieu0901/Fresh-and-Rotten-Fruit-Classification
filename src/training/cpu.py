"""Let a long CPU training job use every core on Windows 11."""
import sys


def use_all_cores():
    """Windows 11 power-saving modes run long background jobs on the efficiency cores only (EcoQoS);
    opt the current process out of that throttling. Does nothing on other systems."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    process = wintypes.HANDLE(kernel32.GetCurrentProcess())
    state = PowerThrottlingState(1, 0x1, 0)  # execution-speed throttling: off
    kernel32.SetProcessInformation(process, 4, ctypes.byref(state), ctypes.sizeof(state))  # 4 = ProcessPowerThrottling
