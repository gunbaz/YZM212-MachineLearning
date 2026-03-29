# Lab 3: Özdeğerler ve Özvektörler

Bu dizin, özdeğer (eigenvalue) ve özvektör (eigenvector) analizini temelden anlamaya yönelik sıfırdan implementasyon ve NumPy fonksiyonlarıyla karşılaştırmaları içerir. 

## 1. Matris Manipülasyonu, Özdeğer ve Özvektörlerin Makine Öğrenmesi ile İlişkisi

Makine öğrenmesinde veriler genellikle çok boyutlu uzaylarda matrisler ve vektörler şeklinde ifade edilir. Matris manipülasyonu (çarpma, ters alma, ayrıştırma vb.) bu veriler üzerinde işlemler yapabilmemizin temelini oluşturur. Ağırlıkların güncellenmesi, veri dönüşümleri ve temel işlemler genellikle lineer cebir temeline dayanır.

**Özdeğer (Eigenvalue)** ve **Özvektör (Eigenvector)**, bir matrisin temel özelliklerini tanımlayan matematiksel kavramlardır. Bir lineer dönüşüm (matris) uygulandığında sadece ölçeği değişen, yönünü değiştirmeyen vektörlere özvektör; bu ölçek faktörüne ise özdeğer denir ($Av = \lambda v$).

Makine öğrenmesindeki en önemli kullanım alanları:
1. **Temel Bileşenler Analizi (PCA - Principal Component Analysis):** Boyut azaltma tekniklerinin en yaygınıdır. Veri setinin kovaryans matrisinin özdeğer ve özvektörleri hesaplanır. En büyük özdeğerlere sahip özvektörler, verideki varyansın (bilginin) en çok korunduğu temel (principal) bileşenleri oluşturur. Böylece yüksek boyutlu veriler, anlam kaybı en aza indirilerek düşük boyutlara indirgenir.
2. **Tekil Değer Ayrışımı (SVD - Singular Value Decomposition):** Kare olmayan matrisler için kavramsal olarak özdeğer ayrışımının daha genel bir halidir. Tavsiye sistemlerinde (Recommender Systems), doğal dil işlemede (LSA/LSI) ve görüntü sıkıştırmada yoğun bir şekilde kullanılır.
3. **Markov Zincirleri ve PageRank:** Birbirine geçiş olasılıkları olan durumları modellerken (örneğin Google'ın PageRank algoritması), geçiş matrisinin en büyük özdeğerine karşılık gelen özvektörü sitelerin önem derecelerini verir.

*Kaynaklar:*
- *Mathematics for Machine Learning - M. P. Deisenroth, A. A. Faisal, C. S. Ong*
- *Deep Learning Book - Ian Goodfellow (Bölüm 2: Linear Algebra)*
- *[Eigenvalues and Eigenvectors in Machine Learning - Towards Data Science](https://towardsdatascience.com/)*

---

## 2. `numpy.linalg.eig` Fonksiyonunun Dokümantasyon ve Kaynak Kod Analizi

NumPy kütüphanesinin `numpy.linalg.eig` fonksiyonu, genel bir karesel matrisin özdeğerlerini ve sağ özvektörlerini hesaplar.

- **Kullanım:** `w, v = numpy.linalg.eig(a)`
  - `w`: Özdeğerleri içeren 1 boyutlu (1D) dizi.
  - `v`: Sütunları özvektörlerden oluşan 2 boyutlu (2D) dizi. `v[:, i]`, `w[i]` özdeğerine karşılık gelen (sağ) özvektördür.

**Kaynak Kod Analizi ve Altyapı:**
NumPy'ın altında yatan asıl güç, `LAPACK` (Linear Algebra PACKage) adlı Fortran kütüphanesidir. `numpy.linalg.eig` Python'da çağrıldığında, arka planda C dilinde yazılmış olan sarmalayıcı (C-API wrapper) kodları aracılığıyla veri matrisinin veri tipine (float32, float64, complex vb.) bağlı olarak spesifik LAPACK rutinleri tetiklenir:
- Reel matrisler için `_geev` rutini (örneğin çift (double) hassasiyet için `dgeev`).
- Karmaşık (complex) matrisler için `zgeev` veya `cgeev` benzeri rutinler.

LAPACK kütüphanesi bu işlemi modern CPU mimarileri (vektörel işlemler, önbellek hizalaması vb.) için son derece optimize edilmiş bir şekilde hesaplar. Algoritma temelde, giriş matrisini önce Hessenberg formuna dönüştürür; ardından Hessenberg formundaki matrise QR algoritması uygulayarak özdeğerleri ve özvektörleri bulur. Bu yapı donanımsal optimizasyonlar sayesinde son derece hızlıdır, sayısal (numerik) kararlılık sunar ve hataları minimize eder.

---

## 3. LucasBN Reposunun Yaklaşımı ile NumPy'ın Karşılaştırması

LucasBN'in *[Eigenvalues-and-Eigenvectors](https://github.com/LucasBN/Eigenvalues-and-Eigenvectors)* adlı GitHub deposunda, özdeğer ve özvektör hesaplama algoritmaları tamamen Python dilinde sıfırdan (from scratch) implement edilmiştir. Bu uygulamada asıl amaç performanstan ziyade eğitici bir model sunmaktır.
- Özdeğerleri hesaplamak için standart **QR Algoritması (Gram-Schmidt iterasyonları)** kullanılmıştır.
- Özvektörleri bulmak için ise **Ters İterasyon (Inverse Iteration / Rayleigh Quotient Iteration)** algoritması uygulanmıştır.

**Karşılaştırma (NumPy `linalg.eig` vs. Sıfırdan Python Yaklaşımı):**

| Özellik | LucasBN (Sıfırdan QR + Ters İterasyon) | NumPy (`numpy.linalg.eig` / LAPACK) |
| :--- | :--- | :--- |
| **Hız** | Düşük. Python `for` döngüleri ve iteratif Gram-Schmidt hesaplamaları performans açısından kötüdür. | Çok Yüksek. C/Fortran tabanlıdır ve donanım BLAS/LAPACK optimizasyonlarını doğrudan kullanır. |
| **Sayısal Kararlılık (Numerical Stability)** | Klasik Gram-Schmidt algoritması, kayan nokta (float) hesaplama duyarlılığı limitlerinden ötürü ortogonallik kaybına (loss of orthogonality) çok yatkındır. Matris boyutu büyüdükçe hata birikir. | Householder yansımaları (reflections) ve blok matris işlemleri kullanır. Float hatalarının önüne geçer, çok daha kesin ve stabil sonuçlar verir. |
| **Genelleştirilmiş Doğruluk** | Algoritma basittir, asimetrik matrislerdeki kompleks özdeğerler (karmaşık sayılar) veya çoklu katlı özdeğerlerde çökebilir, yakınsamayabilir. | Dünyadaki en genel geçer ve kanıtlanmış algoritmaları barındırır. Reel/kompleks, simetrik veya rastgele matrislerde kusursuz başarı oranına sahiptir. |
| **Eğitim Değeri** | Algoritmaların iç dünyasını okuyup lineer cebirin satır satır nasıl derlendiğini (QR ayrışımı) anlamak için muazzamdır. | Kara kutu (black-box) olarak çalışır. Sadece sonucu teslim eder; öğrenmekten veya keşfetmekten ziyade üretim ortamında (production) güvenle kullanabilmek için tasarlanmıştır. |

Özetle, matrislerin ve doğrusal dönüşümlerin temel mekaniğini öğrenmek ve laboratuvar çalışmaları için algoritmaları sıfırdan yazmak (LucasBN'in yaklaşımı) mükemmel bir pratik iken; veri bilimi, modelleme ve üretim (machine learning) uygulamalarında her zaman optimize edilmiş, stabil `NumPy / LAPACK` kütüphaneleri tercih edilmelidir.
