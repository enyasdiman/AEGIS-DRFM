# AEGIS-DRFM: Kapsamlı Mühendislik ve Sistem Tasarım Raporu

## 1. Proje Vizyonu
**AEGIS-DRFM**, bir elektronik harp (EW) mühendisinin simülasyon tasarımından (MATLAB R2025b), algoritma prototiplemeye (Python DSP), gerçek zamanlı yazılım mimarisine (C++20 + SoapySDR) ve donanım tasarımına (SystemVerilog FPGA RTL) kadar ihtiyaç duyduğu tüm zinciri tek bir doğrulanmış ekosistemde birleştirir.

## 2. Modül Bazlı Mühendislik Özeti

### 2.1. MATLAB R2025b Sistem Tasarımı (`matlab/`)
- **Radar Menzil & J/S Bütçesi (`rf_ew_system_design.m`):** 10 GHz X-Band atış kontrol radarı (`Pt = 25 kW`, `B = 20 MHz`, `PW = 10 us`, `PRF = 5 kHz`, `N = 64`) için menzil çözünürlüğü (`7.49 m`), hız çözünürlüğü (`1.17 m/s`), darbe sıkıştırma kazancı (`+23.01 dB`), CPI kazancı (`+18.06 dB`) ve Self-Screening Jammer burn-through menzili hesaplanır ve `system_config.json` üzerinden Python'a aktarılır.
- **DRFM ADC/DAC Kuantalama (`drfm_quantization_sfdr.m`):** 1, 2, 4, 8 ve 12-bit kuantalama için SQNR, ENOB ve harmonik spur seviyeleri (SFDR) modellenir (1-bit: `9.54 dBc`, 8-bit: `67.76 dBc`, 12-bit: `91.00 dBc`).
- **AESA Monopulse Cross-Eye (`aesa_monopulse_crosseye_design.m`):** 16 elemanlı ULA monopulse eğimi (`0.2193 / deg`) ve Cross-Eye açı çekme referans değerleri (`-1.663 deg`, `Quad = 0.0794`) üretilir.

### 2.2. Çekirdek Radar/RF ve DRFM ECM (`packages/core-dsp/`, `packages/drfm-ecm/`)
- LFM, Barker (5/7/11/13), Frank çok fazlı dalgaformları ve Sabit/Staggered/Jittered PRI zamanlayıcıları.
- FFT kesirli gecikme, iki yönlü Doppler, multipath, LO faz gürültüsü, I/Q dengesizliği, Matched Filter, Range-Doppler haritası, 2D CA-CFAR ve 1D OS-CFAR.
- Donanım döngü gecikmeli (`120 ns`) RGPO, VGPO, kinematik kuplajlı RGPO+VGPO, Çoklu Sahte Hedef, ISRJ ve gürültü karıştırma.

### 2.3. SIGINT / ELINT & Bilişsel EW (`packages/sigint-esm/`)
- Geniş bantlı IQ akışında kör darbe tespiti, PDW çıkarımı, darbe içi MOP sınıflandırma (`LFM_UP`, `LFM_DOWN`, `BPSK_BARKER`, `CW`), çok seviyeli **SDIF** deinterleaving ve `CognitiveEWController` ile tehdit önceliğine göre otomatik DRFM ECM görevi.

### 2.4. Çok Katmanlı ECCM (`packages/eccm-radar/`)
- **Dalgaformu Çevikliği:** 1-PRI tekrarlayıcı DRFM'e karşı **+26.3 dB** karıştırma bastırma kazancı.
- **DRFM Parmak İzi:** De-chirp sonrası ters SFDR, spektral düzlük, zarf varyasyonu ve faz jitter öznitelikleriyle 1–4 bit DRFM'de **AUC = 1.000**.
- **Kinematik Kalman İzleyici:** +800 m parabolik RGPO saldırısı altında korumasız `251.3 m` RMS hatayı **`1.4 m`** seviyesine düşürür.

### 2.5. AESA Monopulse & Cross-Eye + SAR Sahte Sahne (`packages/aesa-monopulse/`, `packages/sar-isar/`)
- **Cross-Eye DRFM:** 15 m kanat açıklığı, 3 km menzil, $a=0.90, \phi=178^\circ$ noktasında **11.6x kanat açıklığı** açı sapması + quadrature kanalı ECCM tespiti.
- **SAR TDBP & Sahte Uçak:** 150 MHz bant genişliğiyle 1 m sınıfı 2B SAR odaklama ve +3 dB J/S ile koherent hayalet savaş uçağı enjeksiyonu.

### 2.6. Gerçek Zamanlı C++20, SoapySDR & SystemVerilog FPGA (`packages/rt-cpp-engine/`)
- Lock-free `SpscRingBuffer`, `IRFTransport` / `SoapySDRTransport`, sabit noktalı `DRFMCoreQ15` ve bağımsız (`-static`) `aegis_rt_capi.dll` + Python `ctypes` köprüsü.
- Ölçülen C++20 başarımı: **580–608 MSPS**, **`1.64–1.72 ns/sample`**, Python altın modeliyle **0-LSB bit-exact** eşleşme.
- Zynq/RFSoC/Verilator uyumlu AXI-Stream `drfm_core_q15.sv` RTL + testbench + `$readmemh` hex vektörleri.

## 3. Doğrulama Özeti
- `python -m unittest discover -s tests -v` → **18/18 test OK** (`~0.20s`), 3 bağımsız MATLAB R2025b ↔ Python çapraz doğrulaması ve C++20 DLL bit-exact testi dahil.
