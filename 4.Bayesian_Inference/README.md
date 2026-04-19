# Ödev 4 — Uzak Bir Galaksinin Parlaklık Analizi (Bayesyen Çıkarım)

YZM212 Makine Öğrenmesi dersi 4. Laboratuvar Ödevi

---

## Problem Tanımı

Gürültülü gözlem verilerinden bir gök cisminin gerçek parlaklığını (μ) ve
gözlem belirsizliğini (σ) Bayesyen yöntemle tahmin ediyoruz. Astronomide
deney yapılamadığı için Bayesyen çıkarım — prior bilgi, tam posterior
dağılım ve küçük veri setleriyle iyi çalışma avantajı sayesinde — altın
standart kabul edilir.

---

## Veri

Sentetik olarak üretilmiş 50 gözlemlik parlaklık verisi:

- `true_mu = 150.0` (gerçek parlaklık)
- `true_sigma = 10.0` (gözlem gürültüsü)
- `n_obs = 50`
- `np.random.seed(42)` (deterministik üretim)

Veri `data/gozlem_verisi.csv` içinde kaydedilmiştir.

---

## Yöntem

`emcee` kütüphanesi ile **Markov Chain Monte Carlo (MCMC)** örneklemesi:

- 32 walker, 2000 adım; ilk 500 adım burn-in, `thin=15`.
- **Log-Likelihood:** Gauss varsayımı.
- **Log-Prior:** `0 < μ < 300`, `0 < σ < 50` (geniş, informatif olmayan).
- **Log-Posterior:** Prior + Likelihood (Bayes teoremi).

Kod: [src/bayesian_brightness.ipynb](src/bayesian_brightness.ipynb)

---

## Sonuçlar

### 5.1 · Parametre Karşılaştırma Tablosu

| Değişken | Gerçek Değer | Tahmin (Median) | Alt Sınır (%16) | Üst Sınır (%84) | Mutlak Hata |
|----------|:------------:|:---------------:|:---------------:|:---------------:|:-----------:|
| μ (Parlaklık) | 150.0 | 147.732 | 146.356 | 149.044 | 2.268 |
| σ (Hata Payı) |  10.0 |   9.467 |   8.617 |  10.519 | 0.533 |

### Grafikler

| Grafik | Açıklama |
|--------|----------|
| [veri_histogrami.png](report/veri_histogrami.png) | Sentetik gözlem verisi dağılımı |
| [trace_plot.png](report/trace_plot.png) | MCMC walker zincirleri (yakınsama) |
| [corner_plot.png](report/corner_plot.png) | Posterior dağılımı (μ, σ) |
| [prior_dar_karsilastirma.png](report/prior_dar_karsilastirma.png) | Dar prior etkisi (100–110) |
| [n_obs_5_karsilastirma.png](report/n_obs_5_karsilastirma.png) | Veri miktarı etkisi (n=50 vs n=5) |

Tüm grafikler ve yorumlar: [report/Odev4_Rapor.pdf](report/Odev4_Rapor.pdf)

---

## Yorum / Tartışma

### 6.1. Merkezi Eğilim ve Doğruluk (Accuracy)

Posterior median μ = **147.73**, gerçek değer 150.0 → mutlak hata **2.27** (~%1.5).
Gürültü oranı σ/μ ≈ %6.7 olmasına rağmen MCMC, ortalamaya yakın bir tahminde bulundu.
Model yapısal yanlılık içermiyor; sapma örneklem boyutundan kaynaklanan doğal varyasyon.

### 6.2. Tahmin Hassasiyeti (Precision) Karşılaştırması

Büyük-örneklem teorisinde SE(μ̂) = σ/√n, SE(σ̂) = σ/√(2n).
σ'yı tahmin etmek ikinci moment bilgisi gerektirir → kuyruk verisine duyarlıdır.
n = 50 ile μ için yeterli bilgi birikir; σ için posterior daha geniş ve asimetrik kalır.

### 6.3. Olasılıksal Korelasyon Analizi

Gauss likelihood'unda μ ve σ matematiksel olarak bağımsızdır.
Corner plot'taki 2B konturların eksenlere paralel dik elips olması bunu doğrular —
μ ve σ arasındaki posterior korelasyon ≈ 0.

### Deney A — Dar Prior Etkisi (Soru 1)

μ için prior [100, 110] aralığına daraltıldığında posterior, veriye rağmen prior
sınırına sıkıştı. Yanlış ama güçlü bir prior, veriyi bastırır.
Priorlar fiziksel gerekçelendirmeyle seçilmelidir.

### Deney B — Veri Miktarı Etkisi (Soru 2)

n=50 → n=5'e düşüldüğünde posterior genişliği ~√10 ≈ 3.16 kat arttı.
Bayesyen çıkarım küçük veri setlerinde bile tam bir belirsizlik dağılımı sunar.

---

## Çalıştırma

```bash
pip install -r ../requirements.txt
jupyter notebook src/bayesian_brightness.ipynb
```
