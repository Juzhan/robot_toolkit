"""Voice cues through Windows SAPI or macOS ``say``, off the main control thread.

The Bluetooth keep-alive stream and pre-emption avoid the long silence a
freshly opened audio device otherwise introduces before every cue.

真机反馈驱动的规则：
1. 完整播报：通报串行播放，正在播的内容不会被下一条从中间截断；
   还没开始播的旧消息会被更新的消息替换（最新消息优先，播放队列不积压）。
2. 保持流（默认开，配置 collection.voice_keepalive）：像音乐播放器一样，
   程序运行期间持续向默认音频设备推一段极轻的保持流，蓝牙 A2DP 链路不会
   转入休眠，短语音的开头不再被吞。关掉后回退到"唤醒音 + 静音间隔 +
   冷启动首字重复"的兜底方案。
3. 可抢占：按键确认类消息（开始/取消/拖动开关/中止等，preempt=True）可
   立即切掉正在播的旧内容，保证操作反馈及时；普通状态通报默认排队播完。
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

if sys.platform == 'win32':
    from ctypes import wintypes
else:
    wintypes = None

_AUDIO_DIR = Path(__file__).resolve().parents[1] / 'assets' / 'audio'
_VOICE_KEYS = ('huihui', 'yaoyao', 'kangkang', 'xiaoxiao', 'zh-cn', 'chinese')

_WARM_IDLE_S = 15.0   # 距上次发声超过这么久，播放前重新唤醒音频流（蓝牙防吞字）
_WARM_GAP_S = 0.5     # 唤醒音结束后、语音开始前的静音间隔
_SPEAK_TIMEOUT_S = 60.0

_KEEPALIVE_CHUNK_S = 0.25   # 保持流缓冲块长度（秒）
_KEEPALIVE_RATE = 16000     # 保持流格式：单声道 16bit，约 -90dB 的极轻噪声


if wintypes is not None:
    class _WaveFormatEx(ctypes.Structure):
        _fields_ = [('wFormatTag', wintypes.WORD), ('nChannels', wintypes.WORD),
                    ('nSamplesPerSec', wintypes.DWORD), ('nAvgBytesPerSec', wintypes.DWORD),
                    ('nBlockAlign', wintypes.WORD), ('wBitsPerSample', wintypes.WORD),
                    ('cbSize', wintypes.WORD)]


    class _WaveHdr(ctypes.Structure):
        _fields_ = [('lpData', ctypes.c_void_p), ('dwBufferLength', wintypes.DWORD),
                    ('dwBytesRecorded', wintypes.DWORD), ('dwUser', ctypes.c_void_p),
                    ('dwFlags', wintypes.DWORD), ('dwLoops', wintypes.DWORD),
                    ('lpNext', ctypes.c_void_p), ('reserved', ctypes.c_void_p)]
else:
    _WaveFormatEx = None
    _WaveHdr = None


class _KeepAlive(threading.Thread):
    """持续向默认音频设备推流，让蓝牙链路保持激活（类似音乐播放器）。

    蓝牙耳机空闲一段时间后会休眠 A2DP 链路，下一次短语音的开头会被吞掉。
    只要程序活着就保持一个活跃的播放流，语音即到即播、开头完整。
    """
    BUFFERS = 4
    PUMP_S = 0.05
    _WHDR_DONE = 0x00000001

    def __init__(self):
        super().__init__(daemon=True)
        self._want = threading.Event()
        self._want.set()
        self._stop = threading.Event()
        self.streaming = False
        self._reported = False
        self._winmm = None
        self._handle = None
        self._bufs = []
        self._hdrs = []

    def run(self):
        while not self._stop.is_set():
            if not self._want.is_set():
                self._close()
                if self._stop.wait(0.5):
                    return
                continue
            if not self.streaming and not self._open():
                if not self._reported:
                    print('音频保持流打开失败（稍后自动重试）', flush=True)
                    self._reported = True
                if self._stop.wait(3.0):
                    return
                continue
            try:
                self._pump()
            except Exception:
                self._close()
            if self._stop.wait(self.PUMP_S):
                return
        self._close()

    def _open(self):
        try:
            winmm = ctypes.WinDLL('winmm')
            winmm.waveOutOpen.argtypes = [ctypes.POINTER(wintypes.HANDLE), wintypes.UINT,
                                          ctypes.POINTER(_WaveFormatEx), ctypes.c_void_p,
                                          ctypes.c_void_p, wintypes.DWORD]
            winmm.waveOutOpen.restype = wintypes.UINT
            winmm.waveOutPrepareHeader.argtypes = [wintypes.HANDLE, ctypes.POINTER(_WaveHdr),
                                                   wintypes.UINT]
            winmm.waveOutWrite.argtypes = [wintypes.HANDLE, ctypes.POINTER(_WaveHdr), wintypes.UINT]
            winmm.waveOutUnprepareHeader.argtypes = [wintypes.HANDLE, ctypes.POINTER(_WaveHdr),
                                                     wintypes.UINT]
            winmm.waveOutReset.argtypes = [wintypes.HANDLE]
            winmm.waveOutClose.argtypes = [wintypes.HANDLE]
            fmt = _WaveFormatEx(1, 1, _KEEPALIVE_RATE, _KEEPALIVE_RATE * 2, 2, 16, 0)
            handle = wintypes.HANDLE()
            if winmm.waveOutOpen(ctypes.byref(handle), 0, ctypes.byref(fmt), None, None, 0) != 0:
                return False
            self._winmm = winmm
            self._handle = handle
            chunk = int(_KEEPALIVE_RATE * _KEEPALIVE_CHUNK_S) * 2
            self._bufs, self._hdrs = [], []
            for _ in range(self.BUFFERS):
                buf = ctypes.create_string_buffer(chunk)  # 全零 = 静音
                for offset in range(0, chunk, 2000):
                    buf[offset] = b'\x01'  # 极轻抖动，规避个别链路的静音抑制
                hdr = _WaveHdr()
                hdr.lpData = ctypes.addressof(buf)
                hdr.dwBufferLength = chunk
                if winmm.waveOutPrepareHeader(handle, ctypes.byref(hdr),
                                              ctypes.sizeof(_WaveHdr)) != 0:
                    self._close()
                    return False
                if winmm.waveOutWrite(handle, ctypes.byref(hdr), ctypes.sizeof(_WaveHdr)) != 0:
                    self._close()
                    return False
                self._bufs.append(buf)
                self._hdrs.append(hdr)
            self.streaming = True
            return True
        except Exception:
            self._close()
            return False

    def _pump(self):
        for hdr in self._hdrs:
            if hdr.dwFlags & self._WHDR_DONE:
                if self._winmm.waveOutWrite(self._handle, ctypes.byref(hdr),
                                            ctypes.sizeof(_WaveHdr)) != 0:
                    raise RuntimeError('waveOutWrite 失败')

    def _close(self):
        try:
            if self._winmm is not None and self._handle is not None:
                try:
                    self._winmm.waveOutReset(self._handle)
                except Exception:
                    pass
                for hdr in self._hdrs:
                    try:
                        self._winmm.waveOutUnprepareHeader(self._handle, ctypes.byref(hdr),
                                                           ctypes.sizeof(_WaveHdr))
                    except Exception:
                        pass
                try:
                    self._winmm.waveOutClose(self._handle)
                except Exception:
                    pass
        finally:
            self._bufs, self._hdrs = [], []
            self._handle = None
            self.streaming = False


_keepalive = None
_keepalive_enabled = sys.platform == 'win32'


def set_keepalive(enabled=True):
    """开启/关闭音频保持流（对应配置 collection.voice_keepalive）。"""
    global _keepalive_enabled
    _keepalive_enabled = bool(enabled) and sys.platform == 'win32'
    _ensure_keepalive()


def _ensure_keepalive():
    global _keepalive
    if sys.platform != 'win32':
        return
    if not _keepalive_enabled:
        if _keepalive is not None:
            _keepalive._want.clear()
        return
    if _keepalive is None or not _keepalive.is_alive():
        _keepalive = _KeepAlive()
        _keepalive.start()
    else:
        _keepalive._want.set()


def keepalive_streaming():
    return bool(_keepalive is not None and _keepalive.streaming)


def _wait_keepalive_streaming(timeout):
    if sys.platform != 'win32' or not _keepalive_enabled:
        return False
    _ensure_keepalive()
    deadline = time.perf_counter() + float(timeout)
    while time.perf_counter() < deadline:
        if keepalive_streaming():
            return True
        time.sleep(0.05)
    return keepalive_streaming()


class _Speaker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._preempt = threading.Event()
        self._pending = None
        self._playing = False
        self._last_end = 0.0
        self._winsound = None
        self._wake_path = None
        self._say_proc = None
        self.ready = threading.Event()
        self.engine = None

    def _init(self):
        if sys.platform == 'darwin':
            self.engine = 'say' if shutil.which('say') else None
            self.ready.set()
            return
        try:
            import comtypes
            comtypes.CoInitialize()
            import comtypes.client
            engine = comtypes.client.CreateObject('SAPI.SpVoice')
            voices = engine.GetVoices()
            for index in range(voices.Count):
                voice = voices.Item(index)
                try:
                    desc = voice.GetDescription()
                except Exception:
                    continue
                if any(key in desc.lower() for key in _VOICE_KEYS):
                    engine.Voice = voice
                    break
            self.engine = engine
        except Exception as exc:
            print('\u8bed\u97f3\u5f15\u64ce\u521d\u59cb\u5316\u5931\u8d25\uff1a' + str(exc), flush=True)
            self.engine = None
        try:
            import winsound
            self._winsound = winsound
            wake = _AUDIO_DIR / 'wake.wav'
            self._wake_path = wake if wake.exists() else None
        except Exception:
            self._winsound = None
        finally:
            self.ready.set()

    def _warm_up(self):
        """蓝牙冷启动：先放唤醒音，再留一段静音，避免语音开头被吞掉。"""
        if self._winsound is None or self._wake_path is None:
            return
        try:
            self._winsound.PlaySound(str(self._wake_path), self._winsound.SND_FILENAME)
        except Exception:
            return
        remaining = _WARM_GAP_S
        while remaining > 0 and not self._preempt.is_set():
            step = min(0.05, remaining)
            time.sleep(step)
            remaining -= step

    def _play(self, text):
        if self._preempt.is_set():
            return
        hot = _wait_keepalive_streaming(1.2) if _keepalive_enabled else False
        cold = (sys.platform == 'win32' and not hot
                and (time.perf_counter() - self._last_end) >= _WARM_IDLE_S)
        if cold:
            self._warm_up()
            if self._preempt.is_set():
                return
        if self.engine is None:
            from .audio import beep
            beep(880, 150, block=True)
            return
        spoken = str(text)
        if cold and len(spoken) > 1:
            spoken = spoken[0] + '\uff0c' + spoken  # 冷启动防吞字：重复首字
        if self.engine == 'say':
            try:
                self._say_proc = subprocess.Popen(
                    ['say', '-v', 'Tingting', spoken],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                deadline = time.perf_counter() + _SPEAK_TIMEOUT_S
                timed_out = False
                while self._say_proc.poll() is None:
                    if self._preempt.is_set() or time.perf_counter() > deadline:
                        timed_out = not self._preempt.is_set()
                        self._say_proc.terminate()
                        break
                    time.sleep(0.05)
                try:
                    _, error = self._say_proc.communicate(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._say_proc.kill()
                    _, error = self._say_proc.communicate()
                if not self._preempt.is_set():
                    if timed_out:
                        print('语音播放失败：say 播放超时', flush=True)
                    elif self._say_proc.returncode:
                        detail = (error or b'').decode('utf-8', errors='replace').strip()
                        print(f'语音播放失败：say 返回 {self._say_proc.returncode}：{detail}',
                              flush=True)
            except Exception as exc:
                print(f'语音播放失败：{exc}', flush=True)
            finally:
                self._last_end = time.perf_counter()
            return
        try:
            self.engine.Speak(spoken, 1)  # SAPI 异步播放（不清空队列）
            deadline = time.perf_counter() + _SPEAK_TIMEOUT_S
            while True:
                try:
                    if self.engine.WaitUntilDone(50):
                        break
                except Exception:
                    break
                if self._preempt.is_set():
                    try:
                        self.engine.Speak('', 3)  # 清空并打断当前播报
                    except Exception:
                        pass
                    break
                if time.perf_counter() > deadline:
                    break
        finally:
            self._last_end = time.perf_counter()

    def run(self):
        self._init()
        while True:
            self._wake.wait()
            with self._lock:
                text, self._pending = self._pending, None
                self._preempt.clear()
                self._playing = text is not None
                self._wake.clear()
            if text is None:
                break
            try:
                self._play(text)
            except Exception as exc:
                print('\u8bed\u97f3\u64ad\u653e\u5931\u8d25\uff1a' + str(exc), flush=True)
            finally:
                with self._lock:
                    self._playing = False

    def say(self, text, preempt=False):
        with self._lock:
            self._pending = text
            if preempt and self._playing:
                self._preempt.set()
        self._wake.set()


_speaker = None


def speak(text, preempt=False):
    """Queue a phrase (non-blocking).

    默认：正在播的播完再播新的，尚未开播的旧消息被最新一条替换。
    preempt=True：立即打断正在播的旧内容，用于按键确认类反馈。
    """
    global _speaker
    _ensure_keepalive()
    if _speaker is None:
        _speaker = _Speaker()
        _speaker.start()
    _speaker.say(text, preempt=preempt)
    return _speaker


def wait_ready(timeout=6.0):
    if _speaker is not None:
        _speaker.ready.wait(timeout)
