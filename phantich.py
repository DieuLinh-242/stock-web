import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from dulieu import OUT_DIR, LIMIT, load_stock_data

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

TOL = 1e-6
SHOW = False
DPI_SAVE = 200
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    """Lưu ảnh vào output/, rồi hiện hoặc đóng hình."""
    fig.savefig(OUT_DIR / name, dpi=DPI_SAVE, bbox_inches="tight")
    print(f"[Đã lưu ảnh] {name}")
    if SHOW:
        plt.show()
    plt.close(fig)

def save_csv(df_, name, index=True):
    df_.to_csv(OUT_DIR / name, index=index, encoding="utf-8-sig")
    print(f"[Đã lưu bảng] {name}")

def add_note(fig, text, bottom=0.10):
    """Ghi chú giải thích cách đọc biểu đồ, đặt dưới cùng của hình."""
    fig.tight_layout(rect=[0, bottom, 1, 1])
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=10, color="dimgray")

sd = load_stock_data()
df = pd.DataFrame(sd.prices, columns=[f"Mức_{i}" for i in range(1, sd.k + 1)])
df["Ngày"] = sd.dates
df["Giá"] = sd.close
df["Tháng"] = sd.dates.month
df["Quý"] = sd.dates.quarter
df["Tham_chiếu"] = sd.ref
df["Giá_trần"] = sd.ceil
df["Giá_sàn"] = sd.floor
months = sorted(df["Tháng"].unique())
YEAR = int(sd.dates[0].year)
TOL = 1e-6

def close_to(a, b):
    return np.isclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                      rtol=TOL, atol=TOL)

is_ceil = close_to(df["Giá"], df["Giá_trần"])
is_floor = close_to(df["Giá"], df["Giá_sàn"])
is_ref = close_to(df["Giá"], df["Tham_chiếu"])

#Giá trung bình và độ biến động theo tháng
print("=" * 20, "CÂU 1: THEO THÁNG", "=" * 20)
monthly = df.groupby("Tháng")["Giá"].agg(
    Trung_binh="mean", Do_lech_chuan="std", Thap_nhat="min", Cao_nhat="max")
monthly["Bien_do"] = monthly["Cao_nhat"] - monthly["Thap_nhat"]
print(monthly.round(2))
save_csv(monthly.round(2), "cau1_thang.csv")

m_high, m_low = monthly["Trung_binh"].idxmax(), monthly["Trung_binh"].idxmin()
m_std = monthly["Do_lech_chuan"].idxmax()
m_range = monthly["Bien_do"].idxmax()
print("\nNhận xét:")
print(f"- Giá trung bình cao nhất ở tháng {m_high} ({monthly.loc[m_high, 'Trung_binh']:.2f}), "
      f"thấp nhất ở tháng {m_low} ({monthly.loc[m_low, 'Trung_binh']:.2f}).")
print(f"- Độ lệch chuẩn lớn nhất ở tháng {m_std} ({monthly.loc[m_std, 'Do_lech_chuan']:.2f}) "
      f"=> giá tháng này biến động mạnh nhất quanh mức trung bình.")
print(f"- Tháng có biên độ giá (cao nhất - thấp nhất) lớn nhất: THÁNG {m_range} "
      f"({monthly.loc[m_range, 'Thap_nhat']:.2f} -> {monthly.loc[m_range, 'Cao_nhat']:.2f}, "
      f"biên độ {monthly.loc[m_range, 'Bien_do']:.2f}).")

# Biểu đồ cột: giá trung bình mỗi tháng
fig, ax = plt.subplots(figsize=(12, 6.5))
colors = ["steelblue"] * len(months)
colors[months.index(m_high)] = "seagreen"
colors[months.index(m_low)] = "indianred"
bars = ax.bar(months, monthly["Trung_binh"], color=colors)
ax.bar_label(bars, fmt="%.0f", padding=3)
ax.set_ylim(0, monthly["Trung_binh"].max() * 1.12)
ax.set_xticks(months)
ax.set_xticklabels([f"Tháng {m}" for m in months])
ax.set_ylabel("Giá đóng cửa trung bình")
ax.set_title(f"Giá đóng cửa trung bình của từng tháng, năm {YEAR}", fontsize=13, fontweight="bold")
ax.legend(handles=[Patch(color="seagreen", label=f"Tháng có giá trung bình cao nhất (tháng {m_high})"),
                   Patch(color="indianred", label=f"Tháng có giá trung bình thấp nhất (tháng {m_low})"),
                   Patch(color="steelblue", label="Các tháng còn lại")], loc="upper left")
ax.grid(axis="y", alpha=0.3)
save_fig(fig, "cau1_cot_trung_binh.png")

# Boxplot giá từng tháng
fig, ax = plt.subplots(figsize=(13, 7))
ax.boxplot([df.loc[df["Tháng"] == m, "Giá"] for m in months], showmeans=True, patch_artist=True,
           boxprops=dict(facecolor="lightsteelblue"),
           medianprops=dict(color="darkorange", linewidth=2),
           meanprops=dict(marker="^", markerfacecolor="green", markeredgecolor="green"))
ax.set_xticklabels([f"Tháng {m}" for m in months])
ax.set_ylabel("Giá đóng cửa")
ax.set_title(f"Phân bố giá đóng cửa theo từng tháng (boxplot), năm {YEAR}", fontsize=13, fontweight="bold")
ax.margins(y=0.12)
pos = months.index(m_range) + 1
top = monthly.loc[m_range, "Cao_nhat"]
ax.annotate(f"Tháng {m_range}: biên độ giá\nlớn nhất ({monthly.loc[m_range, 'Bien_do']:.0f})",
            xy=(pos + 0.08, top), xytext=(pos + 1.6, top * 1.02), ha="left", va="center", color="red",
            fontweight="bold", arrowprops=dict(arrowstyle="->", color="red"))
ax.legend(handles=[
    Patch(facecolor="lightsteelblue", edgecolor="black", label="Hộp: 50% số ngày ở giữa (từ Q1 đến Q3)"),
    Line2D([0], [0], color="darkorange", lw=2, label="Vạch cam: trung vị (giá của ngày đứng giữa)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="green", markersize=9, label="Tam giác xanh: giá trung bình"),
    Line2D([0], [0], color="black", lw=1, label="Râu: từ mức giá thấp nhất đến cao nhất (không tính ngoại lệ)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="black",
           label="Vòng tròn: ngày có giá bất thường so với phần còn lại (ngoại lệ)")],
    loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
save_fig(fig, "cau1_boxplot.png")

print("\nGiải thích boxplot: hộp là khoảng từ Q1 đến Q3 (50% số phiên ở giữa), vạch giữa hộp là trung vị, "
      "râu là phạm vi giá thông thường, chấm tròn ngoài râu là giá ngoại lệ. "
      "Hộp/râu càng dài thì giá tháng đó càng dao động mạnh; hộp nằm cao hơn thì mặt bằng giá tháng đó cao hơn.")
iqr = df.groupby("Tháng")["Giá"].agg(lambda s: s.quantile(0.75) - s.quantile(0.25))
print(f"Tháng có hộp rộng nhất (IQR lớn nhất): tháng {iqr.idxmax()} ({iqr.max():.2f}) "
      f"| tháng có hộp hẹp nhất (ổn định nhất): tháng {iqr.idxmin()} ({iqr.min():.2f}).")

#Biến động giá theo quý
print("\n" + "=" * 20, "CÂU 2: THEO QUÝ", "=" * 20)
quarterly = df.groupby("Quý")["Giá"].agg(
    Trung_binh="mean", Do_lech_chuan="std", Thap_nhat="min", Cao_nhat="max")
quarterly["Bien_do"] = quarterly["Cao_nhat"] - quarterly["Thap_nhat"]
quarterly["He_so_bien_dong_%"] = quarterly["Do_lech_chuan"] / quarterly["Trung_binh"] * 100
print(quarterly.round(2))
save_csv(quarterly.round(2), "cau2_quy.csv")

q_cv = quarterly["He_so_bien_dong_%"].idxmax()
q_cv_min = quarterly["He_so_bien_dong_%"].idxmin()
q_rg = quarterly["Bien_do"].idxmax()
print("\nNhận xét:")
print(f"- Quý biến động mạnh nhất (hệ số biến động = độ lệch chuẩn/trung bình): Quý {q_cv} "
      f"({quarterly.loc[q_cv, 'He_so_bien_dong_%']:.2f}%).")
print(f"- Quý ổn định nhất: Quý {q_cv_min} ({quarterly.loc[q_cv_min, 'He_so_bien_dong_%']:.2f}%).")
print(f"- Quý có biên độ giá lớn nhất: Quý {q_rg} ({quarterly.loc[q_rg, 'Bien_do']:.2f}).")

#Ngày bất thường
print("\n" + "=" * 20, "CÂU 3: NGÀY GIAO DỊCH BẤT THƯỜNG", "=" * 20)
df["Lý_do"] = ""
df.loc[is_ceil, "Lý_do"] += "Giá = giá trần; "
df.loc[is_floor, "Lý_do"] += "Giá = giá sàn; "
df.loc[is_ref & ~is_ceil & ~is_floor, "Lý_do"] += "Giá = giá tham chiếu; "

abnormal = df.loc[df["Lý_do"] != "", ["Ngày", "Giá", "Tham_chiếu", "Giá_sàn", "Giá_trần", "Lý_do"]]
abnormal = abnormal.reset_index(drop=True)
if abnormal.empty:
    print("Không có ngày giao dịch nào bất thường theo các tiêu chí trên.")
else:
    abnormal = abnormal.round({"Giá": 2, "Tham_chiếu": 2, "Giá_sàn": 2, "Giá_trần": 2})
    abnormal["Ngày"] = abnormal["Ngày"].dt.date
    print(abnormal)
    print(f"\nTổng cộng {len(abnormal)} ngày bất thường.")
    save_csv(abnormal, "cau3_ngay_bat_thuong.csv", index=False)

#Phân bố giá của 2 quý đầu năm và 2 quý cuối năm
print("\n" + "=" * 20, "CÂU 4: PHÂN BỐ GIÁ 2 QUÝ ĐẦU vs 2 QUÝ CUỐI", "=" * 20)
h1 = df.loc[df["Quý"].isin([1, 2]), "Giá"]
h2 = df.loc[df["Quý"].isin([3, 4]), "Giá"]
bins = np.linspace(df["Giá"].min(), df["Giá"].max(), 21)

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
for ax, data, name, color in [(axes[0], h1, "Nửa đầu năm (Quý 1 + Quý 2)", "tab:blue"),
                              (axes[1], h2, "Nửa cuối năm (Quý 3 + Quý 4)", "tab:orange")]:
    ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(data.mean(), color="red", linestyle="--", linewidth=2, label=f"Giá trung bình = {data.mean():.0f}")
    ax.axvline(data.median(), color="black", linestyle=":", linewidth=2, label=f"Trung vị = {data.median():.0f}")
    ax.set_title(f"{name}\n({len(data)} ngày giao dịch)", fontsize=12)
    ax.set_xlabel("Giá đóng cửa")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Số ngày giao dịch")
fig.suptitle(f"Phân bố giá đóng cửa: nửa đầu năm so với nửa cuối năm {YEAR}", fontsize=14, fontweight="bold")
save_fig(fig, "cau4_histogram.png")

print(f"Nhận xét: 2 quý đầu: TB {h1.mean():.2f}, độ lệch chuẩn {h1.std():.2f}, "
      f"khoảng {h1.min():.2f}-{h1.max():.2f}.")
print(f"          2 quý cuối: TB {h2.mean():.2f}, độ lệch chuẩn {h2.std():.2f}, "
      f"khoảng {h2.min():.2f}-{h2.max():.2f}.")
if h1.mean() != 0:
    print(f"          Giá trung bình nửa cuối/nửa đầu năm: {h2.mean() / h1.mean():.2f} lần.")
print(f"          Trung vị nửa đầu năm: {h1.median():.2f}; trung vị nửa cuối năm: {h2.median():.2f}.")
print(f"          Độ lệch chuẩn nửa đầu năm: {h1.std():.2f}; nửa cuối năm: {h2.std():.2f}.")
print("          Nhận xét lệch phân bố: "
      f"nửa đầu năm {'lệch phải' if h1.skew() > 0.5 else ('lệch trái' if h1.skew() < -0.5 else 'không lệch rõ')}; "
      f"nửa cuối năm {'lệch phải' if h2.skew() > 0.5 else ('lệch trái' if h2.skew() < -0.5 else 'không lệch rõ')}.")

stats4 = pd.DataFrame({
    "Nua_dau_nam (Q1+Q2)": [len(h1), h1.mean(), h1.median(), h1.std(), h1.min(), h1.max()],
    "Nua_cuoi_nam (Q3+Q4)": [len(h2), h2.mean(), h2.median(), h2.std(), h2.min(), h2.max()],
}, index=["So_ngay", "Trung_binh", "Trung_vi", "Do_lech_chuan", "Thap_nhat", "Cao_nhat"]).round(2)
save_csv(stats4, "cau4_thong_ke.csv")

#Xu hướng cả năm
print("\n" + "=" * 20, "CÂU 5: XU HƯỚNG CẢ NĂM", "=" * 20)

daily = sd.daily_frame() 
long = sd.long_frame()

slope, intercept = np.polyfit(sd.days, sd.close, 1)
first, last = sd.close[0], sd.close[-1]
change_pct = (last - first) / first * 100
trend = "TĂNG" if slope > TOL else ("GIẢM" if slope < -TOL else "ĐI NGANG")
print(f"- Giá đầu năm {first:.2f} ({sd.dates[0].date()}) -> cuối năm {last:.2f} "
      f"({sd.dates[-1].date()}): thay đổi {change_pct:+.2f}%.")
print(f"- Đường xu hướng tuyến tính có hệ số góc {slope:.4f}/ngày => xu hướng chung là {trend}.")

i_max, i_min = long["price"].idxmax(), long["price"].idxmin()
p_max, p_min = long.loc[i_max, "price"], long.loc[i_min, "price"]
d_max, d_min = long.loc[i_max, "date"], long.loc[i_min, "date"]
day_max, day_min = long.loc[i_max, "day"], long.loc[i_min, "day"]
ses_max, ses_min = long.loc[i_max, "session"], long.loc[i_min, "session"]
amp = p_max - p_min

print(f"- Giá cao nhất: {p_max:.2f} (ngày {day_max}, {d_max:%d/%m/%Y}, mức giá thứ {ses_max}/{sd.k} trong ngày).")
print(f"- Giá thấp nhất: {p_min:.2f} (ngày {day_min}, {d_min:%d/%m/%Y}, mức giá thứ {ses_min}/{sd.k} trong ngày).")
print(f"- BIÊN ĐỘ DAO ĐỘNG LỚN NHẤT trong năm = {amp:.2f} ({amp / p_min * 100:.2f}% so với mức thấp nhất), "
      f"giữa ngày {min(day_max, day_min)} ({min(d_max, d_min):%d/%m}) và ngày {max(day_max, day_min)} "
      f"({max(d_max, d_min):%d/%m}).")

i_rg = daily["range"].idxmax()
print(f"- Biên độ trong MỘT ngày lớn nhất: {daily.loc[i_rg, 'range']:.2f} "
      f"(ngày {daily.loc[i_rg, 'day']}, {daily.loc[i_rg, 'date']:%d/%m/%Y}).")

n_edge = int((close_to(sd.close, sd.ceil) | close_to(sd.close, sd.floor)).sum())
within_limits = (sd.close <= sd.ceil + TOL * np.maximum(1, np.abs(sd.ceil))) & \
                (sd.close >= sd.floor - TOL * np.maximum(1, np.abs(sd.floor)))
print(f"- Biên độ trần/sàn được đặt ở ±{LIMIT * 100:.0f}% so với giá tham chiếu; "
      f"có {n_edge}/{sd.n} ngày đóng cửa đúng giá trần hoặc sàn.")
if not within_limits.all():
    print(f"  Cảnh báo: {int((~within_limits).sum())} ngày có giá đóng cửa nằm ngoài biên trần/sàn đã tính.")


fig, ax = plt.subplots(figsize=(14, 7))
ax.fill_between(sd.dates, sd.low, sd.high, color="steelblue", alpha=0.25,
                label="Vùng giá trong ngày (thấp nhất - cao nhất)")
ax.plot(sd.dates, sd.close, color="steelblue", linewidth=1.5, label="Giá đóng cửa hằng ngày")
ax.plot(sd.dates, intercept + slope * sd.days, "r--", linewidth=2,
        label=f"Xu hướng chung cả năm ({trend.lower()}, trung bình {slope:+.0f} mỗi ngày)")
ax.scatter([d_max], [p_max], color="green", s=80, zorder=5,
           label=f"Giá cao nhất: {p_max:.0f} (ngày {d_max:%d/%m/%Y})")
ax.scatter([d_min], [p_min], color="red", s=80, zorder=5,
           label=f"Giá thấp nhất: {p_min:.0f} (ngày {d_min:%d/%m/%Y})")
ax.axhline(p_max, color="green", linestyle=":", alpha=0.6)
ax.axhline(p_min, color="red", linestyle=":", alpha=0.6)

x_arrow = sd.dates[-1] + pd.Timedelta(days=6)
ax.annotate("", xy=(x_arrow, p_max), xytext=(x_arrow, p_min),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.8), annotation_clip=False)
ax.text(x_arrow + pd.Timedelta(days=3), (p_max + p_min) / 2, f"Biên độ\n{amp:.0f}",
        va="center", ha="left", fontweight="bold")

ax.margins(y=0.1)
for q in [1, 2, 3, 4]:
    start = daily.loc[daily["quarter"] == q, "date"].min()
    ax.axvline(start, color="gray", linestyle="-", alpha=0.25)
    ax.text(start, ax.get_ylim()[1], f"  Quý {q}", va="top", color="gray", fontsize=10)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
ax.set_ylabel("Giá (VND)")
ax.set_title(f"Diễn biến giá cả năm {YEAR}: xu hướng chung và biên độ dao động lớn nhất",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
save_fig(fig, "cau5_xu_huong.png")

tom_tat = pd.DataFrame({
    "Chỉ tiêu": ["Giá đầu năm", "Giá cuối năm", "Thay đổi (%)", "Hệ số góc xu hướng (/ngày)",
                 "Giá cao nhất", "Ngày giá cao nhất", "Giá thấp nhất", "Ngày giá thấp nhất",
                 "Biên độ dao động lớn nhất"],
    "Giá trị": [round(first, 2), round(last, 2), round(change_pct, 2), round(slope, 4),
                round(p_max, 2), f"{d_max:%d/%m/%Y} (ngày {day_max})",
                round(p_min, 2), f"{d_min:%d/%m/%Y} (ngày {day_min})", round(amp, 2)],
})
save_csv(tom_tat, "cau5_tom_tat.csv", index=False)
print(f"\nXong. Kết quả đã lưu trong: {OUT_DIR}")