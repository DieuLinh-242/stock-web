import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

from dulieu import (BASE, C_CEIL, C_DOWN, C_FLOOR, C_REF, C_TXT, C_UP,
                        StockData, load_stock_data)

MP3_FILE = BASE / "nhac.mp3"
SILENT_MP4 = BASE / "stock_silent.mp4"
FINAL_MP4 = BASE / "stock.mp4"

FPS = 24
DAY_REP = 4
W, H, DPI = 1920, 1080, 100

WINDOW = 60
X_PAD = 4
X_STEP = 10
CANDLE_W = 0.7
Y_SMOOTH = 0.25

ZOOM_SEC = 2.5
HOLD_SEC = 3.0
PRESET = "veryfast"
CRF = 18

def fp(v):
    return f"{v:.6f}"

def lerp(p, q, e):
    return p + (q - p) * e

def x_step_for(span_x):
    for step in (X_STEP, 2 * X_STEP, 2.5 * X_STEP, 5 * X_STEP):
        if span_x / step <= 16:
            return step
    return 5 * X_STEP

def setup_style():
    plt.rcParams.update({
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "xtick.major.width": 2.5, "ytick.major.width": 2.5,
        "xtick.major.size": 8, "ytick.major.size": 8,
    })

class StockVideo:
    def __init__(self, data: StockData):
        self.d = data
        self.day_frames = data.n * DAY_REP
        self.zoom_frames = int(ZOOM_SEC * FPS)
        self.hold_frames = int(HOLD_SEC * FPS)
        self.total_frames = self.day_frames + self.zoom_frames + self.hold_frames
        self.head_end = f"Cả năm: {data.n} ngày"
        self._ylim = {"t": None, "b": None}

        setup_style()
        self._build_figure()
        self._layout_title()

    def _build_figure(self):
        self.fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="black")
        L, R_W = 0.07, 0.80
        self.left_margin = L
        self.ax_top = self.fig.add_axes([L, 0.535, R_W, 0.355], facecolor="black")
        self.ax_bot = self.fig.add_axes([L, 0.095, R_W, 0.355], facecolor="black",
                                        sharex=self.ax_top)

        fmt = FuncFormatter(lambda v, _: f"{v:,.0f}")
        for ax in (self.ax_top, self.ax_bot):
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
            ax.set_ylabel("VND", color=C_TXT, fontsize=14)
            ax.tick_params(colors=C_TXT, labelsize=13)
            for s in ax.spines.values():
                s.set_color("white")
            ax.grid(color="#8B8B8B", linewidth=0.6, alpha=0.9)
        self.ax_top.tick_params(labelbottom=False)
        self.ax_bot.set_xlabel("Ngày", color=C_TXT, fontsize=14)

        self._build_top()
        self._build_bottom()
        self._build_title()

    def _build_top(self):
        ax = self.ax_top
        tr = ax.get_yaxis_transform()
        self.wick_lc = LineCollection([[(0, 0), (0, 0)]], colors=C_UP,
                                      linewidths=2.0, zorder=2)
        self.body_pc = PolyCollection([[(0, 0)] * 4], facecolors=C_UP,
                                      edgecolors=C_UP, linewidths=0.6, zorder=3)
        ax.add_collection(self.wick_lc)
        ax.add_collection(self.body_pc)
        self.cur_line = ax.axhline(0, color=C_UP, ls=":", lw=1.4, zorder=4)
        self.cur_tag = ax.text(1.008, 0, "", transform=tr, color="black",
                               fontsize=14, fontweight="bold", va="center",
                               ha="left", zorder=6,
                               bbox=dict(boxstyle="square,pad=0.25", fc=C_UP, ec="none"))

    def _build_bottom(self):
        ax = self.ax_bot
        d = self.d
        tr = ax.get_yaxis_transform()
        self.fill_pc = PolyCollection([[(0, 0)] * 3], facecolors=C_UP,
                                      edgecolors="none", alpha=0.4, zorder=1)
        ax.add_collection(self.fill_pc)
        (self.ceil_ln,) = ax.plot([], [], color=C_CEIL, ls="--", lw=1.8, zorder=2)
        (self.floor_ln,) = ax.plot([], [], color=C_FLOOR, ls="--", lw=1.8, zorder=2)
        (self.price_ln,) = ax.plot([], [], color=C_UP, lw=2.8, zorder=4,
                                   solid_capstyle="round", solid_joinstyle="round")
        (self.dot,) = ax.plot([], [], "o", ms=6, zorder=5)
        self.price_lbl = ax.annotate(
            "", xy=(0, 0), xytext=(14, 0), textcoords="offset points",
            fontsize=16, fontweight="bold", va="center", ha="left", zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.75))

        self.hline = ax.axhline(d.run_high[0], color=C_UP, ls="--", lw=1.2, alpha=0.9)
        self.lline = ax.axhline(d.run_low[0], color=C_DOWN, ls="--", lw=1.2, alpha=0.9)
        self.htxt = ax.text(1.008, d.run_high[0], "", transform=tr, color=C_UP,
                            fontsize=12, fontweight="bold", va="bottom", ha="left")
        self.ltxt = ax.text(1.008, d.run_low[0], "", transform=tr, color=C_DOWN,
                            fontsize=12, fontweight="bold", va="bottom", ha="left")

    def _build_title(self):
        colors = [C_TXT, C_UP, C_REF, C_FLOOR, C_CEIL]
        self.title_txt = [self.fig.text(0.07, 0.945, "", color=c, fontsize=19,
                                        fontweight="bold") for c in colors]

    def _title_strings(self, i):
        d = self.d
        pct = (d.close[i] / d.ref[i] - 1) * 100
        return [f"Ngày {i + 1}/{d.n}",
                f"Giá: {fp(d.close[i])} ({pct:+.2f}%)",
                f"Giá tham chiếu: {fp(d.ref[i])}",
                f"Giá sàn: {fp(d.floor[i])}",
                f"Giá trần: {fp(d.ceil[i])}"]

    def _layout_title(self, min_gap=28):
        """Chọn cỡ chữ lớn nhất sao cho 5 mục không đè nhau, rồi cố định vị trí x."""
        r = self.fig.canvas.get_renderer()
        left, right = self.left_margin * W, 0.99 * W
        for fs in range(19, 9, -1):
            widths = [0.0] * 5
            for t in self.title_txt:
                t.set_fontsize(fs)
            for i in range(self.d.n):
                for k, (t, sx) in enumerate(zip(self.title_txt, self._title_strings(i))):
                    t.set_text(sx)
                    widths[k] = max(widths[k], t.get_window_extent(r).width)
            self.title_txt[0].set_text(self.head_end)
            widths[0] = max(widths[0], self.title_txt[0].get_window_extent(r).width)
            if sum(widths) + 4 * min_gap <= right - left:
                break
        gap = (right - left - sum(widths)) / 4
        x = left
        for t, wd in zip(self.title_txt, widths):
            t.set_x(x / W)
            t.set_text("")
            x += wd + gap

    def _smooth_ylim(self, key, target):
        cur = self._ylim[key]
        new = target if cur is None else (
            cur[0] + Y_SMOOTH * (target[0] - cur[0]),
            cur[1] + Y_SMOOTH * (target[1] - cur[1]))
        self._ylim[key] = new

    @staticmethod
    def _clamp_label(txt, value, ylim, name):
        lo, hi = ylim
        if value > hi:
            txt.set_position((1.008, hi)); txt.set_va("top")
            txt.set_text(f"▲ {name} {fp(value)}")
        elif value < lo:
            txt.set_position((1.008, lo)); txt.set_va("bottom")
            txt.set_text(f"▼ {name} {fp(value)}")
        else:
            txt.set_position((1.008, value)); txt.set_va("bottom")
            txt.set_text(f"{name} {fp(value)}")

    def _make_candle(self, j, span):
        d = self.d
        o, c, x = d.open[j], d.close[j], d.days[j]
        lo, hi = min(o, c), max(o, c)
        min_body = span * 0.004
        if hi - lo < min_body:
            mid = (hi + lo) / 2
            lo, hi = mid - min_body / 2, mid + min_body / 2
        hw = CANDLE_W / 2
        body = [[x - hw, lo], [x + hw, lo], [x + hw, hi], [x - hw, hi]]
        wick = [[x, d.low[j]], [x, d.high[j]]]
        return body, wick, (C_UP if c >= o else C_DOWN)

    def _draw_top(self, i, a, span_t, s):
        d = self.d
        cands = [self._make_candle(j, span_t) for j in range(a, i + 1)]
        cols = [x[2] for x in cands]
        self.wick_lc.set_segments([np.array(x[1]) for x in cands])
        self.wick_lc.set_color(cols)
        self.wick_lc.set_linewidths(max(0.7, 2.0 * s))
        self.body_pc.set_verts([np.array(x[0]) for x in cands])
        self.body_pc.set_facecolor(cols)
        self.body_pc.set_edgecolor(cols)
        self.body_pc.set_linewidths(max(0.2, 0.6 * s))

        c = C_UP if d.close[i] >= d.ref[i] else C_DOWN
        self.cur_line.set_ydata([d.close[i]] * 2)
        self.cur_line.set_color(c)
        self.cur_tag.set_position((1.008, d.close[i]))
        self.cur_tag.set_text(fp(d.close[i]))
        self.cur_tag.get_bbox_patch().set_facecolor(c)

    def _draw_bottom(self, i, a, ylim_b, s):
        d = self.d
        c = C_UP if d.close[i] >= d.ref[i] else C_DOWN
        xs = d.days[a:i + 1]
        self.ceil_ln.set_data(xs, d.ceil[a:i + 1])
        self.floor_ln.set_data(xs, d.floor[a:i + 1])
        self.price_ln.set_data(xs, d.close[a:i + 1])
        self.ceil_ln.set_linewidth(max(0.9, 1.8 * s))
        self.floor_ln.set_linewidth(max(0.9, 1.8 * s))
        self.price_ln.set_linewidth(max(1.4, 2.8 * s))
        self.price_ln.set_color(c)
        self.fill_pc.set_verts([np.vstack([
            np.column_stack([xs, d.ceil[a:i + 1]]),
            np.column_stack([xs[::-1], d.floor[a:i + 1][::-1]])])])
        self.fill_pc.set_facecolor(c)
        self.fill_pc.set_alpha(0.4)
        self.dot.set_data([d.days[i]], [d.close[i]])
        self.dot.set_markersize(max(4, 6 * s))
        self.dot.set_color(c)
        self.price_lbl.xy = (d.days[i], d.close[i])
        self.price_lbl.set_text(fp(d.close[i]))
        self.price_lbl.set_color(c)

        self.hline.set_ydata([d.run_high[i]] * 2)
        self.lline.set_ydata([d.run_low[i]] * 2)
        self._clamp_label(self.htxt, d.run_high[i], ylim_b, "Cao nhất")
        self._clamp_label(self.ltxt, d.run_low[i], ylim_b, "Thấp nhất")

    def _draw_frame(self, i, a, xlim, ylim_t, ylim_b, head=None):
        span_x = xlim[1] - xlim[0]
        s = min(1.0, (WINDOW + X_PAD) / span_x)        # thu nhỏ nét vẽ khi zoom-out

        self.ax_top.set_xlim(*xlim)
        self.ax_top.set_ylim(*ylim_t)
        self.ax_bot.set_ylim(*ylim_b)
        self.ax_bot.xaxis.set_major_locator(MultipleLocator(x_step_for(span_x)))

        self._draw_top(i, a, ylim_t[1] - ylim_t[0], s)
        self._draw_bottom(i, a, ylim_b, s)

        d = self.d
        for t, sx in zip(self.title_txt, self._title_strings(i)):
            t.set_text(sx)
        if head:
            self.title_txt[0].set_text(head)
        self.title_txt[1].set_color(C_UP if d.close[i] >= d.ref[i] else C_DOWN)

    def _update_days(self, f):
        d = self.d
        i = f // DAY_REP
        a = max(0, i - WINDOW + 1)
        left = max(0.5, i + 1.5 - WINDOW)
        xlim = (left, left + WINDOW + X_PAD)

        if f % DAY_REP == 0:                 # chỉ cập nhật trục y 1 lần/ngày
            lo_t, hi_t = d.low[a:i + 1].min(), d.high[a:i + 1].max()
            pad_t = (hi_t - lo_t) * 0.12 or 1.0
            self._smooth_ylim("t", (lo_t - pad_t, hi_t + pad_t))
            lo_b, hi_b = d.floor[a:i + 1].min(), d.ceil[a:i + 1].max()
            pad_b = (hi_b - lo_b) * 0.06 or 1.0
            self._smooth_ylim("b", (lo_b - pad_b, hi_b + pad_b))

        self._draw_frame(i, a, xlim, self._ylim["t"], self._ylim["b"])

    def _update_zoom(self, f):
        d = self.d
        k = min(f - self.day_frames + 1, self.zoom_frames)
        t = k / self.zoom_frames
        e = t * t * (3 - 2 * t)              # ease in-out

        left0 = max(0.5, d.n + 0.5 - WINDOW)
        x0 = (left0, left0 + WINDOW + X_PAD)
        yt0, yb0 = self._ylim["t"], self._ylim["b"]

        x1 = (0.5, d.n + 0.5 + 6)
        pad_t = (d.high.max() - d.low.min()) * 0.08
        yt1 = (d.low.min() - pad_t, d.high.max() + pad_t)
        pad_b = (d.ceil.max() - d.floor.min()) * 0.05
        yb1 = (d.floor.min() - pad_b, d.ceil.max() + pad_b)

        xlim = (lerp(x0[0], x1[0], e), lerp(x0[1], x1[1], e))
        ylim_t = (lerp(yt0[0], yt1[0], e), lerp(yt0[1], yt1[1], e))
        ylim_b = (lerp(yb0[0], yb1[0], e), lerp(yb0[1], yb1[1], e))
        self._draw_frame(d.n - 1, 0, xlim, ylim_t, ylim_b, head=self.head_end)

    def update(self, f):
        if f == 0:
            self._ylim["t"] = self._ylim["b"] = None
        if f < self.day_frames:
            self._update_days(f)
        else:
            self._update_zoom(f)
        return ()

    def _frame_key(self, f):
        """Các khung có cùng key thì hình giống hệt nhau -> chỉ cần vẽ 1 lần."""
        if f < self.day_frames:
            return ("day", f // DAY_REP)
        if f < self.day_frames + self.zoom_frames - 1:
            return ("zoom", f)
        return ("hold",)

    def render(self):
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
               "-r", str(FPS), "-i", "-", "-an",
               "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
               "-pix_fmt", "yuv420p", str(SILENT_MP4)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        last_key, data, drawn = None, b"", 0
        for f in range(self.total_frames):
            key = self._frame_key(f)
            if key != last_key:
                self.update(f)
                self.fig.canvas.draw()
                data = self.fig.canvas.buffer_rgba().tobytes()
                last_key, drawn = key, drawn + 1
            proc.stdin.write(data)
            if f % 24 == 0:
                print(f"\rRender {f + 1}/{self.total_frames} (đã vẽ {drawn} hình)", end="")
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg báo lỗi khi mã hóa video")
        print(f"\rRender xong {self.total_frames} khung (chỉ vẽ {drawn} hình)      ")


def add_music(total_frames):
    """Ghép nhạc (lặp nếu nhạc ngắn hơn video), fade-out 2 giây cuối."""
    dur = total_frames / FPS
    out = FINAL_MP4
    k = 1
    while True:
        try:
            if out.exists():
                out.unlink()
            break
        except PermissionError:
            out = FINAL_MP4.with_name(f"{FINAL_MP4.stem}_{k}{FINAL_MP4.suffix}")
            k += 1
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(SILENT_MP4),
           "-stream_loop", "-1", "-i", str(MP3_FILE),   # lặp nhạc vô hạn, cắt bằng -t
           "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
           "-b:a", "192k", "-af", f"afade=t=out:st={max(dur - 2, 0):.2f}:d=2",
           "-t", f"{dur:.2f}", str(out)]
    subprocess.run(cmd, check=True)
    SILENT_MP4.unlink(missing_ok=True)
    print("Đã ghi:", out)


if __name__ == "__main__":
    print("Thư mục làm việc:", BASE)
    video = StockVideo(load_stock_data())
    video.render()
    add_music(video.total_frames)
    print("Xong")