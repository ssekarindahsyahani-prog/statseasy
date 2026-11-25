"""
StatEasy - Program Analisis Statistik Otomatis
Python + Streamlit

Jalankan dengan:  streamlit run app.py
"""

import re
import statistics
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ======================================================================
# 1. FUNGSI PEMROSESAN DATA
# ======================================================================
def parse_input(teks: str):
    """Ubah teks input menjadi list angka.

    Pemisah yang didukung: koma, spasi, titik koma, atau baris baru.
    Desimal memakai titik (contoh: 7.5).
    Mengembalikan (list_angka, list_token_tidak_valid).
    """
    token = [t for t in re.split(r"[,\s;]+", teks.strip()) if t]
    angka, invalid = [], []
    for t in token:
        try:
            angka.append(float(t))
        except ValueError:
            invalid.append(t)
    return angka, invalid


def fmt(x: float, digit: int = 2) -> str:
    """Format angka: bulat jika memang bulat, selain itu `digit` desimal."""
    if float(x).is_integer():
        return f"{int(x)}"
    return f"{x:.{digit}f}"


# ======================================================================
# 2. FUNGSI STATISTIK
# ======================================================================
def hitung_statistik(data, sampel=True):
    """Hitung semua statistik deskriptif dan kembalikan sebagai dict."""
    n = len(data)
    arr = np.array(data, dtype=float)

    mean = float(np.mean(arr))
    median = float(np.median(arr))

    # Modus: bisa lebih dari satu (multimodal) atau tidak ada
    hitung = Counter(data)
    frek_maks = max(hitung.values())
    if frek_maks == 1:
        modus = []  # semua nilai unik -> tidak ada modus
    else:
        modus = sorted(k for k, v in hitung.items() if v == frek_maks)

    # Varians & standar deviasi (sampel: n-1, populasi: n)
    if sampel and n < 2:
        varians = float("nan")
        std = float("nan")
    else:
        ddof = 1 if sampel else 0
        varians = float(np.var(arr, ddof=ddof))
        std = float(np.std(arr, ddof=ddof))

    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    batas_bawah = q1 - 1.5 * iqr
    batas_atas = q3 + 1.5 * iqr
    outlier = [x for x in data if x < batas_bawah or x > batas_atas]

    cv = (std / mean * 100) if (mean != 0 and not np.isnan(std)) else float("nan")

    return {
        "n": n,
        "jumlah": float(np.sum(arr)),
        "mean": mean,
        "median": median,
        "modus": modus,
        "frek_modus": frek_maks,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "varians": varians,
        "std": std,
        "cv": cv,
        "outlier": outlier,
        "jenis": "sampel" if sampel else "populasi",
    }


# ======================================================================
# 3. INTERPRETASI OTOMATIS
# ======================================================================
def buat_interpretasi(s):
    """Hasilkan daftar kalimat interpretasi sederhana dari hasil statistik."""
    hasil = []

    # --- Pemusatan data ---
    hasil.append(
        f"Rata-rata data adalah **{fmt(s['mean'])}**, artinya jika seluruh nilai "
        f"dibagi rata, setiap data bernilai sekitar {fmt(s['mean'])}. "
        f"Nilai tengahnya (median) adalah **{fmt(s['median'])}**."
    )

    # --- Bentuk sebaran (skewness sederhana: bandingkan mean vs median) ---
    rentang = s["range"] if s["range"] != 0 else 1
    selisih = (s["mean"] - s["median"]) / rentang
    if abs(selisih) < 0.05:
        hasil.append(
            "Mean dan median hampir sama, sehingga data cenderung **simetris** "
            "(tidak condong ke satu sisi)."
        )
    elif selisih > 0:
        hasil.append(
            "Mean lebih besar dari median, sehingga data cenderung **condong ke "
            "kanan** (ada beberapa nilai besar yang menarik rata-rata ke atas)."
        )
    else:
        hasil.append(
            "Mean lebih kecil dari median, sehingga data cenderung **condong ke "
            "kiri** (ada beberapa nilai kecil yang menarik rata-rata ke bawah)."
        )

    # --- Modus ---
    if not s["modus"]:
        hasil.append("Tidak ada **modus** karena semua nilai muncul hanya satu kali.")
    elif len(s["modus"]) == 1:
        hasil.append(
            f"Nilai yang paling sering muncul (modus) adalah **{fmt(s['modus'][0])}** "
            f"sebanyak {s['frek_modus']} kali."
        )
    else:
        daftar = ", ".join(fmt(m) for m in s["modus"])
        hasil.append(
            f"Data bersifat **multimodus**: nilai {daftar} sama-sama muncul "
            f"paling sering ({s['frek_modus']} kali)."
        )

    # --- Penyebaran ---
    if np.isnan(s["std"]):
        hasil.append(
            "Standar deviasi sampel tidak dapat dihitung karena data hanya satu. "
            "Tambahkan minimal 2 data."
        )
    else:
        hasil.append(
            f"Standar deviasi ({s['jenis']}) sebesar **{fmt(s['std'])}** menunjukkan "
            f"bahwa data rata-rata menyimpang sekitar {fmt(s['std'])} dari nilai rata-ratanya."
        )
        if not np.isnan(s["cv"]):
            cv = abs(s["cv"])
            if cv < 10:
                level = "sangat homogen (data seragam)"
            elif cv < 20:
                level = "cukup homogen"
            elif cv < 30:
                level = "cukup bervariasi"
            else:
                level = "sangat bervariasi (heterogen)"
            hasil.append(
                f"Koefisien variasi sebesar **{fmt(cv)}%**, sehingga data tergolong {level}."
            )

    # --- Outlier ---
    if s["outlier"]:
        daftar = ", ".join(fmt(o) for o in s["outlier"])
        hasil.append(
            f"Terdapat **{len(s['outlier'])} data pencilan (outlier)** berdasarkan "
            f"metode IQR: {daftar}. Periksa apakah nilai tersebut benar atau salah input."
        )
    else:
        hasil.append("Tidak ditemukan data pencilan (outlier) berdasarkan metode IQR.")

    return hasil


# ======================================================================
# 4. GRAFIK
# ======================================================================
def plot_histogram(data, bins, mean, median):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=bins, color="#4C78A8", edgecolor="white")
    ax.axvline(mean, color="#E45756", linestyle="--", linewidth=2, label=f"Mean = {fmt(mean)}")
    ax.axvline(median, color="#54A24B", linestyle="-.", linewidth=2, label=f"Median = {fmt(median)}")
    ax.set_title("Histogram")
    ax.set_xlabel("Nilai")
    ax.set_ylabel("Frekuensi")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_bar_data(data, mean):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(1, len(data) + 1)
    ax.bar(x, data, color="#4C78A8", edgecolor="white")
    ax.axhline(mean, color="#E45756", linestyle="--", linewidth=2, label=f"Mean = {fmt(mean)}")
    ax.set_title("Diagram Batang (Nilai per Data)")
    ax.set_xlabel("Urutan data ke-")
    ax.set_ylabel("Nilai")
    if len(data) <= 30:
        ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_bar_frekuensi(data):
    hitung = Counter(data)
    kunci = sorted(hitung)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([fmt(k) for k in kunci], [hitung[k] for k in kunci],
           color="#F58518", edgecolor="white")
    ax.set_title("Diagram Batang (Frekuensi Tiap Nilai)")
    ax.set_xlabel("Nilai")
    ax.set_ylabel("Frekuensi")
    ax.yaxis.get_major_locator().set_params(integer=True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


# ======================================================================
# 5. RINGKASAN (teks yang bisa diunduh)
# ======================================================================
def buat_ringkasan(data, s, interpretasi):
    modus = ", ".join(fmt(m) for m in s["modus"]) if s["modus"] else "Tidak ada"
    bersih = [re.sub(r"\*\*", "", i) for i in interpretasi]
    baris = [
        "=== RINGKASAN ANALISIS STATISTIK (StatEasy) ===",
        f"Data              : {', '.join(fmt(x) for x in data)}",
        f"Jumlah data (n)   : {s['n']}",
        f"Jenis perhitungan : {s['jenis']}",
        "",
        f"Jumlah            : {fmt(s['jumlah'])}",
        f"Mean              : {fmt(s['mean'], 4)}",
        f"Median            : {fmt(s['median'], 4)}",
        f"Modus             : {modus}",
        f"Minimum / Maksimum: {fmt(s['min'])} / {fmt(s['max'])}",
        f"Range             : {fmt(s['range'])}",
        f"Q1 / Q3 / IQR     : {fmt(s['q1'])} / {fmt(s['q3'])} / {fmt(s['iqr'])}",
        f"Varians           : {fmt(s['varians'], 4)}",
        f"Standar deviasi   : {fmt(s['std'], 4)}",
        "",
        "Interpretasi:",
    ]
    baris += [f"- {i}" for i in bersih]
    return "\n".join(baris)


# ======================================================================
# 6. ANTARMUKA STREAMLIT
# ======================================================================
def main():
    st.set_page_config(page_title="StatEasy", page_icon="📊", layout="wide")

    st.title("📊 StatEasy")
    st.caption("Program analisis statistik otomatis — tanpa hitung manual.")

    # ---------------- Sidebar: input ----------------
    with st.sidebar:
        st.header("1. Input Data")
        teks = st.text_area(
            "Masukkan angka (pisahkan dengan koma, spasi, atau baris baru)",
            value="70, 85, 90, 75, 85, 60, 95, 85, 80, 70",
            height=150,
            help="Gunakan titik untuk desimal, contoh: 7.5",
        )
        jenis = st.radio(
            "Jenis perhitungan varians & std",
            ["Sampel (n−1)", "Populasi (n)"],
            help="Gunakan 'Sampel' jika data hanya sebagian dari populasi.",
        )
        bins = st.slider("Jumlah bin histogram", 3, 30, 8)
        tombol = st.button("🔍 Analisis Data", type="primary", width="stretch")

    # ---------------- Validasi ----------------
    data, invalid = parse_input(teks)

    if invalid:
        st.warning(f"Input berikut bukan angka dan diabaikan: {', '.join(invalid)}")

    if not data:
        st.info("Masukkan data angka di panel kiri, lalu klik **Analisis Data**.")
        return

    if not tombol and "sudah_analisis" not in st.session_state:
        st.info("Klik **Analisis Data** di panel kiri untuk memulai.")
        return
    st.session_state["sudah_analisis"] = True

    # ---------------- Hitung ----------------
    s = hitung_statistik(data, sampel=jenis.startswith("Sampel"))
    interpretasi = buat_interpretasi(s)

    # ---------------- Metrik utama ----------------
    st.subheader("Hasil Perhitungan")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Jumlah Data (n)", s["n"])
    k2.metric("Mean", fmt(s["mean"]))
    k3.metric("Median", fmt(s["median"]))
    k4.metric("Modus", ", ".join(fmt(m) for m in s["modus"]) if s["modus"] else "Tidak ada")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric(f"Varians ({s['jenis']})", fmt(s["varians"], 4) if not np.isnan(s["varians"]) else "-")
    k6.metric(f"Std. Deviasi ({s['jenis']})", fmt(s["std"], 4) if not np.isnan(s["std"]) else "-")
    k7.metric("Minimum", fmt(s["min"]))
    k8.metric("Maksimum", fmt(s["max"]))

    # ---------------- Tab ----------------
    tab_grafik, tab_tabel, tab_interp, tab_ringkas = st.tabs(
        ["📈 Grafik", "🧮 Tabel Data", "💡 Interpretasi", "📝 Ringkasan"]
    )

    with tab_grafik:
        c1, c2 = st.columns(2)
        with c1:
            fig = plot_histogram(data, bins, s["mean"], s["median"])
            st.pyplot(fig)
        with c2:
            fig = plot_bar_data(data, s["mean"])
            st.pyplot(fig)
        fig = plot_bar_frekuensi(data)
        st.pyplot(fig)

    with tab_tabel:
        df = pd.DataFrame({"No": range(1, len(data) + 1), "Nilai": data})
        df["Selisih dari mean"] = df["Nilai"] - s["mean"]
        df["Kuadrat selisih"] = df["Selisih dari mean"] ** 2
        st.dataframe(df, width="stretch", hide_index=True)

        st.markdown("**Statistik lengkap**")
        lengkap = pd.DataFrame(
            {
                "Ukuran": ["Jumlah", "Range", "Q1", "Q3", "IQR", "Koef. Variasi (%)"],
                "Nilai": [
                    s["jumlah"], s["range"], s["q1"], s["q3"], s["iqr"],
                    s["cv"] if not np.isnan(s["cv"]) else None,
                ],
            }
        )
        st.dataframe(lengkap, width="stretch", hide_index=True)

    with tab_interp:
        for kalimat in interpretasi:
            st.markdown(f"- {kalimat}")

    with tab_ringkas:
        ringkasan = buat_ringkasan(data, s, interpretasi)
        st.code(ringkasan, language="text")
        st.download_button(
            "⬇️ Unduh ringkasan (.txt)",
            data=ringkasan,
            file_name="ringkasan_statseasy.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
