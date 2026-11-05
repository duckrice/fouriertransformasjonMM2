"""Spektrogram: ta opp en lyd (eller velg et eksempel) og se korttids-fouriertransformasjonen (STFT) av den.

Kjør:  python spektrogram.py
"""
import io
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import wave
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

RECORDING = Path(__file__).with_name("opptak_piano.wav")  # the recording that comes with the program


# ---------- Fourier transform ----------
def fft(x):
    """Cooley-Tukey radix-2 FFT (N log N). len(x) must be a power of 2.

    Column c of x.reshape(m, -1) holds every (N/m)-th sample, so one matrix product does the small
    m-point DFTs of all those "every k-th sample" sub-sequences at once. Then each round merges pairs:
    X_k = E_k + e^(-iπk/M)·O_k and X_(k+M) = E_k - e^(-iπk/M)·O_k, doubling the length M each time.
    """
    x = np.asarray(x, dtype=complex)
    n = x.shape[-1]
    assert n & (n - 1) == 0, "length must be a power of 2"
    m = min(n, 32)
    k = np.arange(m)
    X = np.exp(-2j * np.pi * np.outer(k, k) / m) @ x.reshape(*x.shape[:-1], m, n // m)
    while X.shape[-2] < n:
        size = X.shape[-2]
        half = X.shape[-1] // 2
        even = X[..., :half]
        odd = X[..., half:] * np.exp(-1j * np.pi * np.arange(size) / size)[:, None]
        X = np.concatenate([even + odd, even - odd], axis=-2)
    return X.reshape(x.shape)


SHAPES = {  # one period on x in [0, 1), values between -1 and 1
    "square": lambda x: np.sign(0.5 - x % 1) * (x % 1 != 0),  # 0 at the jumps: the midpoint the series converges to
    "sawtooth": lambda x: 2 * (x - np.floor(x + 0.5)),
    "triangle": lambda x: 1 - 4 * np.abs(x - np.floor(x + 0.5)),
}


def pcm_to_float(raw, width, big_endian, unsigned_8bit):
    """Turn raw PCM bytes into numbers between -1 and 1."""
    if width == 1:
        data = np.frombuffer(raw, np.uint8 if unsigned_8bit else np.int8).astype(float)
        return (data - 128) / 128 if unsigned_8bit else data / 128
    if width == 3:  # 24-bit has no numpy type: build the integers from bytes
        b = np.frombuffer(raw[: len(raw) // 3 * 3], np.uint8).reshape(-1, 3).astype(np.int32)
        if big_endian:
            b = b[:, ::-1]
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        return np.where(ints >= 2**23, ints - 2**24, ints) / 2**23
    if width in (2, 4):
        return np.frombuffer(raw, (">" if big_endian else "<") + f"i{width}") / 2 ** (8 * width - 1)
    raise ValueError(f"Unsupported sample size ({width * 8}-bit).")


def read_audio(path):
    """Read a WAV file as (sample rate, mono samples between -1 and 1)."""
    data = Path(path).read_bytes()
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        try:
            with wave.open(str(path), "rb") as w:
                rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
                raw = w.readframes(w.getnframes())
        except wave.Error as e:
            raise ValueError(f"Unsupported WAV file ({e}). Export it as 16-bit PCM WAV.") from e
        samples = pcm_to_float(raw, width, big_endian=False, unsigned_8bit=True)
    else:
        raise ValueError("Not a WAV file.")
    if len(samples) < channels:
        raise ValueError("The file has no sound in it.")
    return rate, samples[: len(samples) // channels * channels].reshape(-1, channels).mean(axis=1)  # stereo -> mono


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(f):
    """Nearest piano note and how far off it is, e.g. 'A4 +2¢' for 440.5 Hz (A4 = 440 Hz)."""
    if f <= 0:
        return ""
    semis = 12 * np.log2(f / 440)
    k = int(round(semis))
    cents = round(100 * (semis - k))
    return f"{NOTE_NAMES[(k + 9) % 12]}{(k + 57) // 12}" + (f" {cents:+d}¢" if cents else "")


def write_wav(path, rate, signal):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())


def stft(signal, frame=4096):
    """Short-time Fourier transform: the FFT of short overlapping slices (each faded in and out by a Hann window).

    The sound is padded with one slice of silence at both ends so its edges are covered too.
    Returns (slice centres in samples, complex spectra[slice, frequency]).
    """
    hop = frame // 4
    padded = np.concatenate([np.zeros(frame), signal, np.zeros(frame)])
    starts = np.arange(0, len(padded) - frame + 1, hop)
    slices = padded[starts[:, None] + np.arange(frame)] * np.hanning(frame)
    return starts + frame / 2 - frame, fft(slices)[:, : frame // 2 + 1]


def spectrogram(signal, rate, frame=4096):
    """How the spectrum changes over time. Returns (times s, freqs Hz, amplitudes[freq, time]).

    frame sets the trade-off: a slice of frame samples lasts frame/rate seconds, and its frequency
    steps are rate/frame Hz. Sharper in time means blurrier in frequency (time-frequency uncertainty).
    """
    centres, spectra = stft(signal, frame)
    amps = 2 * np.abs(spectra) / np.hanning(frame).sum()
    return centres / rate, np.arange(frame // 2 + 1) * rate / frame, amps.T




# ---------- playing and recording sound ----------
WINDOWS = sys.platform == "win32"
MAC = sys.platform == "darwin"

if WINDOWS:
    import ctypes
    import winsound

    MIC_HELP = ("Check that a microphone is connected, and that Windows lets desktop apps use it:\n"
                "Settings → Privacy & security → Microphone → Let desktop apps access your microphone.")
else:
    MIC_HELP = ("Check that a microphone is connected and not muted, and that it is chosen as the input device\n"
                "(for example in pavucontrol, or run  wpctl status  in a terminal to list the devices).")


def find_program(*names):
    """The first of these command-line programs that is installed, or None."""
    return next((name for name in names if shutil.which(name)), None)


# ---------- playback ----------
_player = None  # the running pw-play / paplay / afplay process (Linux and macOS)


def play_file(path):
    """Start playing a WAV file and return at once. Stops whatever was playing before."""
    global _player
    stop_playback()
    if WINDOWS:
        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        return
    program = "afplay" if MAC else find_program("pw-play", "paplay", "aplay")
    if program is None:
        raise OSError("No sound player found. Install PipeWire (pw-play) or PulseAudio (paplay).")
    _player = subprocess.Popen([program, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_playback():
    """Stop playback (does nothing if nothing is playing)."""
    global _player
    if WINDOWS:
        winsound.PlaySound(None, 0)
    elif _player is not None:
        if _player.poll() is None:
            _player.terminate()
            try:
                _player.wait(timeout=1)
            except subprocess.TimeoutExpired:
                _player.kill()
        _player = None


# ---------- recording ----------
def mci(command):
    """Send a command to Windows' built-in audio interface (MCI). Used for recording, so no extra install is needed."""
    reply = ctypes.create_unicode_buffer(256)
    err = ctypes.windll.winmm.mciSendStringW(command, reply, 255, 0)
    if err:
        msg = ctypes.create_unicode_buffer(256)
        ctypes.windll.winmm.mciGetErrorStringW(err, msg, 255)
        raise OSError(msg.value or f"MCI error {err}")
    return reply.value


class Recorder:
    """Records the microphone (mono, 16 bit, 44100 Hz) to a WAV file: start(path) … stop() … close()."""

    def __init__(self):
        self.process = None  # pw-record / parecord on Linux
        self.path = None

    def start(self, path):
        """Start recording into `path`. Raises OSError if the microphone can't be opened."""
        self.path = path
        if WINDOWS:
            try:
                mci("close rec")  # left open by a crash earlier
            except OSError:
                pass
            mci("open new type waveaudio alias rec")
            mci("set rec bitspersample 16 samplespersec 44100 channels 1 bytespersec 88200 alignment 2")
            mci("record rec")
            return
        if MAC:
            raise OSError("Recording isn't supported on macOS yet. Record with another program and open the WAV file.")
        program = find_program("pw-record", "parecord")
        if program is None:
            raise OSError("No recording program found. Install PipeWire (pw-record) or PulseAudio (parecord).")
        args = ([program, "--rate", "44100", "--channels", "1", "--format", "s16", str(path)] if program == "pw-record"
                else [program, "--rate=44100", "--channels=1", "--format=s16le", "--file-format=wav", str(path)])
        self.process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        time.sleep(0.2)  # a program that can't open the microphone quits right away
        if self.process.poll() is not None:
            error = self.process.stderr.read().strip()
            self.process = None
            raise OSError(error or f"{program} stopped at once.")

    def stop(self):
        """Stop recording and finish writing the WAV file."""
        if WINDOWS:
            mci("stop rec")
            mci(f'save rec "{self.path}"')
            return
        if self.process is not None:
            self.process.send_signal(signal.SIGINT)  # like Ctrl+C: the program finishes the WAV header and quits
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def close(self):
        """Give the microphone back, even after an error."""
        if WINDOWS:
            try:
                mci("close rec")
            except OSError:
                pass
        elif self.process is not None:
            self.process.kill()
            self.process = None


# Fonts: Windows' own, or the first one installed on Linux/macOS (DejaVu comes with matplotlib, so it always exists).
UI_FONTS = ["Segoe UI", "Inter", "Noto Sans", "Cantarell", "Helvetica", "DejaVu Sans"]


# matplotlib warns about every listed font that's missing, so only list the installed ones.
MPL_FONTS = [f for f in UI_FONTS if f in {font.name for font in matplotlib.font_manager.fontManager.ttflist}]


def own_theme(root, style, p, font):
    """A flat theme built on Tk's 'clam' theme, so the window looks the same on every computer."""
    style.theme_use("clam")
    bold = (font[0], font[1], "bold")
    style.configure(".", background=p["bg"], foreground=p["text"], fieldbackground=p["field"], font=font,
                    bordercolor=p["edge"], lightcolor=p["bg"], darkcolor=p["bg"], troughcolor=p["stripe"],
                    selectbackground=p["select"], selectforeground=p["on_select"], insertcolor=p["text"],
                    focuscolor=p["select"], arrowcolor=p["text"])
    style.map(".", foreground=[("disabled", p["muted"])])
    for name in ("TButton", "TMenubutton"):
        style.configure(name, background=p["button"], bordercolor=p["edge"], lightcolor=p["button"],
                        darkcolor=p["button"], padding=(8, 3), relief="flat")
        style.map(name, background=[("pressed", p["edge"]), ("active", p["hover"])],
                  lightcolor=[("active", p["hover"])], darkcolor=[("active", p["hover"])])
    style.configure("TButton", width=0)  # clam makes every button at least 11 letters wide
    style.configure("Accent.TButton", background=p["select"], foreground=p["on_select"], bordercolor=p["select"],
                    lightcolor=p["select"], darkcolor=p["select"], font=bold)
    style.map("Accent.TButton", background=[("active", p["text"])], lightcolor=[("active", p["text"])],
              darkcolor=[("active", p["text"])], foreground=[("active", p["bg"])])
    style.configure("TSpinbox", fieldbackground=p["field"], background=p["button"], padding=2)
    style.configure("TEntry", fieldbackground=p["field"], padding=2)
    style.configure("TNotebook", background=p["bg"], bordercolor=p["edge"], tabmargins=(0, 2, 0, 0))
    style.configure("TNotebook.Tab", background=p["bg"], foreground=p["muted"], padding=(12, 4),
                    bordercolor=p["edge"], lightcolor=p["bg"])
    style.map("TNotebook.Tab", background=[("selected", p["stripe"])], foreground=[("selected", p["text"])],
              lightcolor=[("selected", p["select"])], expand=[("selected", (0, 0, 0, 0))])
    style.configure("Treeview", background=p["field"], fieldbackground=p["field"], foreground=p["text"],
                    rowheight=int(2.2 * font[1]) + 4, bordercolor=p["edge"])
    style.map("Treeview", background=[("selected", p["select"])], foreground=[("selected", p["on_select"])])
    style.configure("Treeview.Heading", background=p["stripe"], foreground=p["muted"], relief="flat", font=bold,
                    bordercolor=p["edge"], lightcolor=p["stripe"], darkcolor=p["stripe"])
    style.map("Treeview.Heading", background=[("active", p["hover"])])
    style.configure("Horizontal.TScale", background=p["select"], troughcolor=p["edge"], bordercolor=p["bg"],
                    lightcolor=p["select"], darkcolor=p["select"])
    style.configure("TProgressbar", background=p["select"], troughcolor=p["stripe"], bordercolor=p["bg"],
                    lightcolor=p["select"], darkcolor=p["select"])
    style.configure("TSeparator", background=p["edge"])
    style.configure("Sash", sashthickness=6, gripcount=0, background=p["bg"])
    style.configure("TCheckbutton", indicatorbackground=p["field"], indicatorforeground=p["select"])
    root.configure(background=p["bg"])
    menu = dict(background=p["stripe"], foreground=p["text"], activebackground=p["select"],
                activeforeground=p["on_select"], relief="flat", borderwidth=0, activeborderwidth=0,
                selectcolor=p["select"], font=font)
    for option, value in menu.items():  # menus opened later pick these up
        root.option_add(f"*Menu.{option}", value)

    def recolour(widget):  # plain Tk widgets (menus, toplevels) don't follow ttk styles
        for child in widget.winfo_children():
            if isinstance(child, tk.Menu):
                child.configure(**menu)
            elif isinstance(child, tk.Toplevel):
                child.configure(background=p["bg"])
            recolour(child)
    recolour(root)


def pick_font(choices):
    """The first font in `choices` that Tk can find on this computer (needs a Tk window to exist)."""
    installed = set(tkfont.families())
    return next((name for name in choices if name in installed), choices[-1])

def example(name, rate=44100):
    t = np.arange(int(2.5 * rate)) / rate
    if name == "sine":
        return 0.5 * np.sin(2 * np.pi * 220 * t)
    if name == "square":
        return 0.5 * np.sign(np.sin(2 * np.pi * 220 * t))
    if name in ("sawtooth", "triangle"):
        return 0.5 * SHAPES[name](220 * t)
    raise ValueError(name)


EXAMPLES = [("sine", "Sine wave (220 Hz)"), ("square", "Square wave (220 Hz)"),
            ("sawtooth", "Sawtooth wave (220 Hz)"), ("triangle", "Triangle wave (220 Hz)")]


# ---------- look ----------
REPORT = dict(bg="#ffffff", axes="#ffffff", text="#1c1c1c", muted="#6b6b6b", edge="#c8c8c8")  # saved figures
# The window's colours (Catppuccin Latte and Mocha).
# field = text boxes, button/hover = buttons, select = highlight colour, on_select = text on it.
CATPPUCCIN = {
    "light": dict(bg="#ffffff", axes="#ffffff", text="#4c4f69", muted="#6c6f85", edge="#bcc0cc",
                  stripe="#f2f3f6", accent="#fe640b", field="#ffffff", button="#e6e9ef", hover="#dce0e8",
                  select="#1e66f5", on_select="#ffffff"),
    "dark": dict(bg="#1e1e2e", axes="#181825", text="#cdd6f4", muted="#a6adc8", edge="#45475a",
                 stripe="#24273a", accent="#fab387", field="#181825", button="#313244", hover="#45475a",
                 select="#89b4fa", on_select="#11111b"),
}
PALETTE = CATPPUCCIN
BLUE = "#3d85c6"


def mpl_style(dark, report=False):
    """matplotlib settings for the graphs. A report export always uses plain white."""
    p = REPORT if report else PALETTE["dark" if dark else "light"]
    return {"figure.facecolor": p["bg"], "axes.facecolor": p["axes"], "savefig.facecolor": p["bg"],
            "axes.edgecolor": p["edge"], "axes.labelcolor": p["text"], "text.color": p["text"],
            "xtick.color": p["muted"], "ytick.color": p["muted"], "xtick.labelcolor": p["text"],
            "ytick.labelcolor": p["text"], "legend.labelcolor": p["text"], "axes.spines.top": False,
            "axes.spines.right": False, "font.family": MPL_FONTS, "font.size": 9,
            "axes.titlesize": 10, "axes.labelsize": 9}


def make_icon():
    """A white sine wave on a blue square, for the title bar and taskbar."""
    size = 32
    img = tk.PhotoImage(width=size, height=size)
    img.put(BLUE, to=(0, 0, size, size))
    for x in range(3, size - 3):
        y = int(round(size / 2 - 9 * np.sin(2 * np.pi * (x - 3) / (size - 6))))
        img.put("white", to=(x, y - 1, x + 1, y + 2))
    return img




TOP_HZ = 5000  # highest frequency shown at first; zoom with the magnifier to see more or less


def plot_spectrogram(ax, spec, top, db_range):
    times, freqs, amps = spec
    db = 20 * np.log10(amps + 1e-12)
    half_hop, half_bin = (times[1] - times[0]) / 2 if len(times) > 1 else 0, freqs[1] / 2
    image = ax.imshow(db, origin="lower", aspect="auto", cmap="magma", vmin=db.max() - db_range, vmax=db.max(),
                      interpolation="nearest",  # imshow: much faster than pcolormesh for big images
                      extent=[times[0] - half_hop, times[-1] + half_hop, -half_bin, freqs[-1] + half_bin])
    ax.set_ylim(0, top)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    return image


class ZoomBar(NavigationToolbar2Tk):
    """matplotlib's toolbar with only Home, Pan and Zoom. Back/Forward are greyed out most of the time and Tk
    draws a greyed-out icon as a grey block on a dark background; saving is in the File menu (Ctrl+S)."""
    toolitems = [t for t in NavigationToolbar2Tk.toolitems if t[0] in ("Home", "Pan", "Zoom")]


class App:
    def __init__(self, root):
        self.root = root
        root.title("Spectrogram")
        root.geometry("1100x700")
        root.minsize(700, 450)
        self.tmp = tempfile.TemporaryDirectory()  # recordings and playback files, deleted on exit
        self.sound_no = 0
        self.job = 0  # bumps on every new spectrogram, so a slow old one can't overwrite a newer result
        self.results = queue.Queue()
        self.name = self.rate = self.signal = self.spec = self.ax = None
        self.spec_frame = None  # slice length the shown spectrogram was made with
        self.zoomed_sound = None  # the sound whose zoom the graph shows; a new sound starts unzoomed
        self.recorder = Recorder()
        self.recording_since = None
        self.dark = tk.BooleanVar(value=False)
        self.quiet = tk.BooleanVar(value=False)
        self.build_menu()
        self.build_ribbon()
        self.build_statusbar()
        self.build_body()
        root.iconphoto(True, make_icon())
        for key, cmd in [("<Control-r>", self.toggle_record), ("<Control-s>", self.save_spectrogram),
                         ("<Control-o>", self.open_file), ("<Control-c>", self.copy_spectrogram),
                         ("<space>", self.play_sound), ("<Escape>", self.stop)]:
            root.bind(key, lambda e, c=cmd: c())
        self.apply_theme()
        self.poll()

    # ---------- layout ----------
    def build_menu(self):
        bar = tk.Menu(self.root)
        file = tk.Menu(bar, tearoff=0)
        file.add_command(label="Record from microphone", accelerator="Ctrl+R", command=self.toggle_record)
        file.add_command(label="Open recording…", accelerator="Ctrl+O", command=self.open_file)
        file.add_cascade(label="Examples", menu=self.examples_menu(file))
        file.add_separator()
        file.add_command(label="Copy spectrogram", accelerator="Ctrl+C", command=self.copy_spectrogram)
        file.add_command(label="Save spectrogram for report…", accelerator="Ctrl+S", command=self.save_spectrogram)
        file.add_command(label="Save sound as WAV…", command=self.save_sound)
        file.add_separator()
        file.add_command(label="Exit", command=self.root.destroy)
        bar.add_cascade(label="File", menu=file)
        view = tk.Menu(bar, tearoff=0)
        view.add_checkbutton(label="Show quiet sounds", variable=self.quiet, command=self.draw)
        view.add_checkbutton(label="Dark mode", variable=self.dark, command=self.apply_theme)
        bar.add_cascade(label="View", menu=view)
        self.root.config(menu=bar)

    def examples_menu(self, parent):
        menu = tk.Menu(parent, tearoff=0)
        if RECORDING.exists():
            menu.add_command(label="Recording (piano)", command=lambda: self.open_file(RECORDING))
            menu.add_separator()
        for item in EXAMPLES:
            if item is None:
                menu.add_separator()
            else:
                key, label = item
                menu.add_command(label=label, command=lambda k=key, l=label: self.load(l, 44100, example(k)))
        return menu

    def menubutton(self, parent, text, **kw):
        button = ttk.Menubutton(parent, text=text, **kw)
        button["menu"] = self.examples_menu(button)
        return button

    def build_ribbon(self):
        ribbon = ttk.Frame(self.root, padding=(8, 6))
        ribbon.pack(fill="x")
        self.rec_btn = ttk.Button(ribbon, text="● Record", style="Accent.TButton", command=self.toggle_record)
        self.rec_btn.pack(side="left")
        self.menubutton(ribbon, "Examples").pack(side="left", padx=4)
        ttk.Separator(ribbon, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(ribbon, text="▶ Play", command=self.play_sound).pack(side="left")
        ttk.Button(ribbon, text="■ Stop", command=self.stop).pack(side="left", padx=4)
        ttk.Separator(ribbon, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(ribbon, text="Copy", command=self.copy_spectrogram).pack(side="left")
        ttk.Button(ribbon, text="Save for report…", command=self.save_spectrogram).pack(side="left", padx=4)
        ttk.Separator(self.root, orient="horizontal").pack(fill="x")

    def build_statusbar(self):
        bar = ttk.Frame(self.root, padding=(8, 3))
        bar.pack(fill="x", side="bottom")  # packed before the body, so it can never be squeezed out
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", side="bottom")
        self.status = tk.StringVar(value="Record a sound or pick an example.")
        ttk.Label(bar, textvariable=self.status).pack(side="left")
        self.readout = tk.StringVar()
        ttk.Label(bar, textvariable=self.readout, style="Muted.TLabel").pack(side="right")
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=120)

    def build_body(self):
        graphs = ttk.Frame(self.root)
        graphs.pack(fill="both", expand=True)
        self.fig = Figure(figsize=(9, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, graphs)
        bottom = ttk.Frame(graphs, padding=(8, 2, 0, 2))
        bottom.pack(side="bottom", fill="x")
        self.toolbar = ZoomBar(self.canvas, bottom, pack_toolbar=False)
        self.toolbar.set_message = lambda s: None  # the status bar already shows time, Hz and note
        if hasattr(self.toolbar, "_message_label"):  # it stretches to fill the row, pushing our controls out
            self.toolbar._message_label.pack_forget()
        self.toolbar.pack_propagate(True)  # matplotlib makes it as wide as the figure, hiding the slider row
        self.toolbar.configure(width=1)
        self.toolbar.pack(side="right")  # packed first, so a narrow window squeezes the text, not the zoom buttons
        ttk.Label(bottom, text="Slice length:", style="Muted.TLabel").pack(side="left")
        ttk.Label(bottom, text="sharp in time").pack(side="left", padx=(6, 0))
        self.frame_pow = tk.DoubleVar(value=12)
        frame_scale = ttk.Scale(bottom, from_=8, to=15, variable=self.frame_pow, length=130, command=self.on_frame_move)
        frame_scale.pack(side="left", padx=4)
        frame_scale.bind("<ButtonRelease-1>", lambda e: self.make_spectrogram())
        frame_scale.bind("<KeyRelease>", lambda e: self.make_spectrogram())
        ttk.Label(bottom, text="sharp in frequency").pack(side="left")
        self.frame_text = tk.StringVar()
        ttk.Label(bottom, textvariable=self.frame_text, style="Muted.TLabel").pack(side="left", padx=10)
        self.on_frame_move()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("motion_notify_event", self.on_move)

        # Start screen, shown over the graph until there is a sound.
        self.welcome = ttk.Frame(graphs, padding=16)
        ttk.Label(self.welcome, text="Spectrogram", style="Title.TLabel").pack(pady=(0, 14))
        ttk.Button(self.welcome, text="● Record from the microphone", style="Accent.TButton", width=28,
                   command=self.toggle_record).pack(pady=3)
        self.menubutton(self.welcome, "Try an example", width=26).pack(pady=3)

    # ---------- look ----------
    def apply_theme(self):
        """Light or dark: the ttk theme, our label styles, the graph and the zoom toolbar."""
        mode = "dark" if self.dark.get() else "light"
        p = PALETTE[mode]
        style = ttk.Style()
        base = (pick_font(UI_FONTS), 10)
        own_theme(self.root, style, p, base)
        colours = {"Muted.TLabel": p["muted"]}

        def recolour(widget):  # ttk labels ignore a style's colour here
            for child in widget.winfo_children():
                if isinstance(child, ttk.Label) and str(child.cget("style")) in colours:
                    child.configure(foreground=colours[str(child.cget("style"))])
                recolour(child)
        recolour(self.root)
        style.configure("Title.TLabel", font=(base[0], 22, "bold"))
        matplotlib.rcParams.update(mpl_style(self.dark.get()))
        self.canvas.get_tk_widget().configure(background=p["bg"])
        for button in getattr(self.toolbar, "_buttons", {}).values():  # icons follow the button background
            button.configure(background=p["bg"], activebackground=p["stripe"], highlightbackground=p["bg"])
            try:
                self.toolbar._set_image_for_button(button)
            except Exception:  # private matplotlib API: plain icons are fine if it changes
                pass
        self.toolbar.configure(background=p["bg"])
        for child in self.toolbar.winfo_children():  # its spacer is a plain Tk label, light grey by default
            if isinstance(child, tk.Label):
                child.configure(background=p["bg"], foreground=p["text"])
        self.draw()

    # ---------- recording ----------
    def toggle_record(self):
        if self.recording_since is None:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        stop_playback()  # don't record our own playback
        self.sound_no += 1
        try:
            self.recorder.start(Path(self.tmp.name) / f"recording{self.sound_no}.wav")
        except OSError as e:
            self.close_recorder()
            messagebox.showerror("Can't record", f"{e}\n\n{MIC_HELP}")
            return
        self.recording_since = time.time()
        self.rec_btn.config(text="■ Stop recording")
        self.tick_recording()

    def tick_recording(self):
        if self.recording_since is not None:
            self.status.set(f"● Recording… {time.time() - self.recording_since:.1f} s. "
                            "Click Stop recording (Ctrl+R) when you're done.")
            self.root.after(200, self.tick_recording)

    def stop_recording(self):
        try:
            self.recorder.stop()
            rate, signal = read_audio(self.recorder.path)
        except (OSError, ValueError) as e:
            messagebox.showerror("Recording failed", f"{e}\n\n{MIC_HELP}")
            return
        finally:
            self.close_recorder()
        if np.abs(signal).max() < 1e-4:
            messagebox.showwarning("The recording is silent", MIC_HELP)
        self.load(f"Recording {time.strftime('%H.%M.%S')}", rate, signal)

    def close_recorder(self):
        self.recorder.close()
        self.recording_since = None
        self.rec_btn.config(text="● Record")

    def open_file(self, path=None):
        path = path or filedialog.askopenfilename(filetypes=[("Sound files", "*.wav *.aif *.aiff"), ("All files", "*.*")])
        if not path:
            return
        try:
            rate, signal = read_audio(path)
        except (OSError, ValueError) as e:
            messagebox.showerror("Can't open file", f"{Path(path).name} couldn't be opened.\n\n{e}")
            return
        self.load(Path(path).stem, rate, signal)

    # ---------- the spectrogram ----------
    def load(self, name, rate, signal):
        self.name, self.rate = name, rate
        self.signal = signal - signal.mean()  # remove the 0 Hz offset some microphones have
        self.spec = None
        self.root.title(f"{name} – Spectrogram")
        self.on_frame_move()
        self.make_spectrogram()

    def frame(self):
        return 2 ** int(round(self.frame_pow.get()))

    def on_frame_move(self, _value=None):
        self.frame_pow.set(round(self.frame_pow.get()))  # snap to a power of two
        rate, frame = self.rate or 44100, self.frame()
        slices = "" if self.signal is None else f"{(len(self.signal) + frame) // (frame // 4) + 1} slices  ·  "  # as in stft()
        self.frame_text.set(f"{frame} samples ({1000 * frame / rate:.0f} ms) per slice  ·  {slices}"
                            f"frequency step {rate / frame:.1f} Hz")

    def make_spectrogram(self):
        if self.signal is None or (self.spec is not None and self.spec_frame == self.frame()):
            return
        self.job += 1
        job, signal, rate, frame = self.job, self.signal, self.rate, self.frame()
        self.status.set(f"Making the spectrogram with {frame}-sample slices…")
        self.progress.pack(side="left", padx=8)
        self.progress.start(15)

        def work():  # in a thread so the window doesn't freeze; tkinter may only be touched from poll()
            try:
                self.results.put((job, (frame, spectrogram(signal, rate, frame))))
            except Exception as e:  # never leave the progress bar spinning forever
                self.results.put((job, e))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        while not self.results.empty():
            job, result = self.results.get()
            if job != self.job:
                continue
            self.progress.stop()
            self.progress.pack_forget()
            if isinstance(result, Exception):
                self.status.set("Making the spectrogram failed.")
                messagebox.showerror("Spectrogram failed", str(result))
                continue
            self.spec_frame, self.spec = result
            self.status.set(f"{self.name}: {self.frame_text.get()}")
            self.draw()
        self.root.after(100, self.poll)

    def db_range(self):
        return 100 if self.quiet.get() else 70

    def draw(self):
        # Keep the zoom when the same sound is drawn again (new slice length, dark mode, quiet sounds).
        keep = (self.ax.get_xlim(), self.ax.get_ylim()) if self.ax is not None and self.zoomed_sound is self.signal else None
        self.zoomed_sound = None
        self.fig.clear()
        self.fig.set_facecolor(matplotlib.rcParams["figure.facecolor"])
        self.ax = None
        if self.signal is None:
            self.welcome.place(relx=0.5, rely=0.45, anchor="center")
            self.canvas.draw_idle()
            return
        self.welcome.place_forget()
        self.ax = self.fig.add_subplot()
        self.ax.set_title(self.name, loc="left", fontsize=10)
        if self.spec is None:
            self.ax.text(0.5, 0.5, "Working…", transform=self.ax.transAxes, ha="center",
                         color=matplotlib.rcParams["xtick.color"])
        else:
            image = plot_spectrogram(self.ax, self.spec, min(TOP_HZ, self.rate / 2), self.db_range())
            self.fig.colorbar(image, ax=self.ax, label="Loudness (dB)", pad=0.02)
        self.ax.set_xlim(0, len(self.signal) / self.rate)
        self.fig.tight_layout()
        self.toolbar.update()  # forget the old graph's zoom history...
        self.toolbar.push_current()  # ...so the house button goes back to the whole spectrogram
        if self.spec is not None:
            self.zoomed_sound = self.signal
            if keep:
                self.ax.set(xlim=keep[0], ylim=keep[1])
        self.canvas.draw_idle()

    def on_move(self, event):
        if event.inaxes is self.ax and self.ax is not None and event.ydata is not None and self.spec is not None:
            # Snap to the tile under the cursor: show what the STFT computed there, not the mouse position.
            times, freqs, amps = self.spec
            k = min(max(round(event.ydata / freqs[1]), 1), len(freqs) - 1)
            m = int(np.argmin(np.abs(times - event.xdata)))
            db = 20 * np.log10(amps[k, m] / amps.max() + 1e-12)
            self.readout.set(f"slice m={m} ({times[m]:.2f} s) · bin k={k}: {freqs[k]:.1f} ± {freqs[1] / 2:.1f} Hz"
                             f" · {note_name(freqs[k])} · {db:.0f} dB")
        else:
            self.readout.set("")

    # ---------- sound ----------
    def play_sound(self):
        if self.signal is not None and np.abs(self.signal).max() > 0:
            self.play(0.8 * self.signal / np.abs(self.signal).max())
            self.status.set(f"Playing {self.name}.")

    def play(self, signal):
        # 20 ms fade in/out: a sound that starts or stops mid-wave jumps, and a jump contains every frequency (a click)
        k = np.arange(len(signal))
        signal = signal * np.minimum(1, np.minimum(k, len(signal) - 1 - k) / (0.02 * self.rate))
        self.sound_no += 1  # new file each time: the one playing may still be open
        path = Path(self.tmp.name) / f"sound{self.sound_no}.wav"
        write_wav(path, self.rate, signal)
        try:
            play_file(path)
        except OSError as e:
            messagebox.showerror("Can't play sound", str(e))

    def stop(self):
        stop_playback()
        self.status.set("Stopped.")

    # ---------- saving ----------
    def save_sound(self):
        if self.signal is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV sound", "*.wav")],
                                            initialfile=f"{self.stem()}.wav")
        if path:
            write_wav(path, self.rate, self.signal)
            self.status.set(f"Saved {Path(path).name}.")

    def stem(self):
        return re.sub(r"[^A-Za-z0-9-]+", "_", self.name).strip("_") or "lyd"

    def save_spectrogram(self):
        if self.spec is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf"), ("PNG image", "*.png")],
                                            initialfile=f"{self.stem()}_spektrogram.pdf")
        if path:
            self.save_spectrogram_to(Path(path))
            self.status.set(f"Saved {Path(path).name} (white, Norwegian labels, the part you're zoomed in on).")

    def copy_spectrogram(self):
        """Put the shown area on the clipboard as a PNG image, ready to paste (Wayland: wl-copy, X11: xclip)."""
        if self.spec is None:
            return
        image = io.BytesIO()
        self.save_spectrogram_to(image, format="png")
        program = find_program("wl-copy", "xclip")
        if program is None:
            messagebox.showerror("Can't copy", "Copying images needs wl-copy (Wayland) or xclip (X11). "
                                               "Use Save for report instead.")
            return
        args = ["wl-copy", "--type", "image/png"] if program == "wl-copy" else \
            ["xclip", "-selection", "clipboard", "-t", "image/png"]
        try:
            subprocess.run(args, input=image.getvalue(), check=True, timeout=5)
        except (OSError, subprocess.SubprocessError) as e:
            messagebox.showerror("Can't copy", str(e))
            return
        self.status.set("Copied the shown area of the spectrogram. Paste it with Ctrl+V.")

    def save_spectrogram_to(self, path, **savefig_options):
        """The spectrogram as a figure for the report: white, Norwegian labels, the zoomed view from the window."""
        with matplotlib.rc_context({**mpl_style(dark=False, report=True), "figure.facecolor": "white",
                                    "savefig.facecolor": "white"}):
            fig = Figure(figsize=(6.5, 3.6))
            ax = fig.add_subplot()
            image = plot_spectrogram(ax, self.spec, TOP_HZ, self.db_range())
            ax.set(xlim=self.ax.get_xlim(), ylim=self.ax.get_ylim(), xlabel="Tid (s)", ylabel="Frekvens (Hz)")
            fig.colorbar(image, ax=ax, label="Lydstyrke (dB)", pad=0.02)
            fig.tight_layout()
            fig.savefig(path, dpi=200, **savefig_options)


if __name__ == "__main__":
    root = tk.Tk(className="Spectrogram")  # window class, so the desktop can match its menu entry and icon
    app = App(root)
    if len(sys.argv) > 1:
        app.open_file(sys.argv[1])
    root.mainloop()
    stop_playback()
    app.recorder.close()
    app.tmp.cleanup()
