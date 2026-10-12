# Mô hình 1 – ARIMA cho tỷ giá USD/VND

Thư mục này chứa riêng phần ARIMA (Mô hình 1, mục 7.1 của báo cáo nhóm): một notebook, dữ liệu đầu vào, hình, bảng kết quả và báo cáo docx. Thư mục chạy độc lập, không cần các notebook khác.

- **Dữ liệu:** tỷ giá Close USD/VND theo phiên, 04/01/2016 – 31/08/2026, 2.784 phiên, đơn vị VND cho 1 USD.
- **Mục tiêu:** dự báo tỷ giá phiên kế tiếp theo quy trình Box–Jenkins, so với Naive (random walk: giá phiên sau bằng giá phiên trước).
- **Mô hình cuối:** ARIMA(3,1,1) trên y = 100 · ln(Close), không có hằng số.

---

## 1. Cấu trúc thư mục

```
ARIMA_USD_VND/
├── README.md
├── phu_luc_A_mo_hinh_1_arima.csv        4 dòng Phụ lục A của Mô hình 1 để chép vào báo cáo nhóm
├── code/
│   └── 05_mo_hinh_arima_usd_vnd.ipynb    Notebook duy nhất, chạy từ đầu đến cuối
├── data/
│   ├── raw/USD_VND_history.csv           Dữ liệu gốc (chỉ để tham chiếu, notebook không đọc)
│   └── processed/
│       ├── price_train.csv               04/01/2016 – 31/12/2024, 2.347 phiên
│       ├── price_validation.csv          01/01/2025 – 31/12/2025, 262 phiên
│       ├── price_test.csv                01/01/2026 – 31/08/2026, 175 phiên
│       └── return_train.csv              Chỉ dùng để kiểm tra diff(100·log_price) = log_return
├── figures/                              7 hình do notebook tạo
├── results/                              19 bảng CSV do notebook tạo
└── report/
    └── Bao_cao_7.1_ARIMA_USD_VND_v4.docx Báo cáo mục 7.1 (Bảng P1.1–P1.11, Hình P1.1–P1.7)
```

Các file trong `data/processed/` lấy nguyên từ bước tiền xử lý và chia tập chung của nhóm (notebook 01–02 ở thư mục `USD_VND_notebooks/`).

## 2. Cách chạy

- Python 3.10 trở lên với `pandas`, `numpy`, `scipy`, `matplotlib`, `statsmodels`. Bộ Anaconda đã có sẵn.
- Notebook dùng đường dẫn tương đối: tự nhận thư mục gốc `ARIMA_USD_VND/`, dù chạy từ `code/` hay từ thư mục gốc.

```bash
# Từ thư mục ARIMA_USD_VND/, mất khoảng 3–4 phút
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 code/05_mo_hinh_arima_usd_vnd.ipynb
```

Hoặc mở notebook trong Jupyter / VS Code và chọn **Run All**. Notebook ghi đè `figures/` và `results/`; kết quả lặp lại được (bootstrap dùng seed 20261010).

## 3. Nội dung notebook

| Mục | Nội dung | Đầu ra | Trong báo cáo |
|---|---|---|---|
| 1 | Đọc dữ liệu, tạo y = 100 · ln(Close) | – | 7.1.1 |
| 2 | ACF/PACF; ADF và KPSS cho d = 0 và d = 1 | `10_arima_identification.png`, `arima_stationarity_tests.csv` | 7.1.2, Hình P1.1 |
| 3 | Kiểm tra hằng số drift | – | 7.1.2 |
| 4 | Lưới AIC/BIC p, q ∈ {0,…,4}; tập 14 ứng viên | `arima_order_selection.csv` | Bảng P1.1 |
| 5 | Kiểm tra overfitting (thêm bậc vào AR(1), MA(1)) | `arima_overfitting.csv` | 7.1.3 |
| 6–7 | Dự báo rolling one-step; so sánh 14 ứng viên trên Validation | – | 7.1.3 |
| 8 | Quy trình chọn mô hình cuối | `arima_validation_metrics.csv`, `22_arima_validation_candidates.png` | Bảng P1.2, Hình P1.2 |
| 9 | Tham số và nghiệm AR/MA của (3,1,1) và (3,1,4); kiểm tra overfitting cho (3,1,1) bằng (4,1,1) và (3,1,2) | `arima_params.csv`, `arima_params_aic.csv`, `arima_overfitting_final.csv` | Bảng P1.3, P1.4 |
| 10 | Hình chẩn đoán phần dư trên Train | `11_arima_residual_diagnostics_final.png` | Hình P1.6 |
| 11 | Đánh giá theo giao thức nhóm: MAE, RMSE, MAPE, sMAPE, MASE, DA, độ phủ, interval score, DM HAC, bootstrap khối; dự báo từng phiên trên Test | `arima_metrics_group_protocol.csv`, `arima_loss_tests_hac_bootstrap.csv`, `arima_test_forecasts.csv`, `15_arima_onestep_test.png` | Bảng P1.5–P1.7, Hình P1.4 |
| 12 | Ước lượng lại trên Train + Validation; Ljung–Box trừ bậc tự do; chẩn đoán sai số Test | `arima_refit_summary.csv`, `arima_residual_tests_df_adjusted.csv`, `arima_test_error_diagnostics.csv` | Bảng P1.8 |
| 13–14 | Dự báo nhiều bước từ mốc cố định 31/12/2025; chênh lệch loss tích lũy | `arima_fixed_origin_multistep.csv`, `20_arima_fixed_origin_test.png`, `21_arima_cumulative_loss_diff_test.png` | Bảng P1.9, Hình P1.5, P1.7 |
| 15 | Độ vững: ước lượng cuốn chiếu, biến giả ngoại lai, ngày trong tuần, σ cuốn chiếu, bảng ba tập, bỏ ngày lễ | `arima_robustness_*.csv`, `arima_metrics_all_sets_ss_format.csv`, `23_arima_relative_three_sets.png` | Bảng P1.10, Hình P1.3 |
| 16 | Kết luận | – | 7.1.10 |

## 4. Thiết kế và lựa chọn tham số

### 4.1. Chia dữ liệu và cách dự báo

| Tập | Thời gian | Số phiên | Dùng để |
|---|---|---|---|
| Train | 04/01/2016 – 31/12/2024 | 2.347 | Nhận dạng, chọn bậc, ước lượng tham số |
| Validation | 01/01/2025 – 31/12/2025 | 262 | Chọn mô hình cuối trong các ứng viên |
| Test | 01/01/2026 – 31/08/2026 | 175 | Đánh giá cuối cùng, chỉ dùng một lần |

- Validation: tham số ước lượng trên Train. Test: tham số ước lượng lại trên Train + Validation rồi giữ cố định.
- Mỗi phiên chỉ dùng thông tin đến phiên trước (rolling one-step, `res.apply()` của statsmodels).
- Dự báo đổi về giá theo kỳ vọng log-normal: P̂ = exp(m/100 + v/(2·100²)), chỉ chênh 0,02–0,03 VND so với trung vị exp(m/100).

### 4.2. Biến và bậc sai phân
- y = 100 · ln(Close): sai phân bậc một chính là log-return tính theo %.
- **d = 1:**

| Chuỗi | ADF p-value | KPSS (thống kê; p) | Kết luận |
|---|---|---|---|
| log-price (d = 0) | 0,908 | 5,118; < 0,01 | Không dừng |
| Sai phân bậc 1 (d = 1) | < 0,001 | 0,140; > 0,10 | Dừng |

- **Không dùng hằng số drift:** return trung bình 0,0054%/phiên (p = 0,049), không ổn định giữa các năm (2021 giảm 1,1%, 2024 tăng 5,0%).

### 4.3. Chọn p, q
- ACF/PACF của sai phân vượt ngưỡng ±0,0405 ở lag 1, 3, 4, 6 và không cắt cụt rõ, nên chọn bậc bằng lưới tiêu chí thông tin.
- AIC chọn ARIMA(3,1,4), BIC chọn ARIMA(3,1,1).
- **Ứng viên:** 5 mô hình tốt nhất theo AIC, 5 tốt nhất theo BIC, cộng (0,1,0), (1,1,0), (0,1,1), (1,1,1), tổng 14 cấu hình.
- **Quy tắc chọn** (cố định trước khi xem Test):
  1. Giữ mô hình hội tụ, mọi nghiệm AR/MA ngoài đường tròn đơn vị, và Ljung–Box phần dư Train (trừ bậc tự do p + q) có p > 0,05 ở lag 10 và 20.
  2. Trong các mô hình hợp lệ có RMSE tỷ giá Validation ≤ 1,01 × mức tốt nhất, chọn p + q nhỏ nhất; hòa thì lấy RMSE nhỏ hơn.

| Mô hình | p + q | RMSE Val (VND) | Hội tụ | LB(10) / LB(20) p | Kết quả |
|---|---|---|---|---|---|
| (0,1,0) Naive | 0 | 43,25 | Có | < 0,001 / < 0,001 | Loại (Ljung–Box) |
| (1,1,1) | 2 | 43,17 | Có | < 0,001 / < 0,001 | Loại (Ljung–Box) |
| (3,1,3) | 6 | 42,90 | Không | 0,34 / 0,53 | Loại (không hội tụ) |
| (3,1,4) | 7 | 42,95 | Có | 0,90 / 0,90 | Hợp lệ, RMSE tốt nhất |
| (4,1,0) | 4 | 43,17 | Có | 0,07 / 0,24 | Hợp lệ, trong ngưỡng 1% |
| **(3,1,1)** | **4** | **43,13** | Có | 0,16 / 0,39 | **Được chọn** |

- Ngưỡng 1% là 43,38 VND. (3,1,1) và (4,1,0) cùng p + q = 4; (3,1,1) có RMSE thấp hơn.
- Không chọn (3,1,4) vì nghiệm AR và MA gần triệt tiêu (môđun 1,375 và 1,362), 4/7 hệ số ARMA không có ý nghĩa, và không hội tụ khi ước lượng lại. Mô hình này được giữ làm đối chứng cùng (1,1,1).

### 4.4. Tham số ARIMA(3,1,1)

| Tham số | φ₁ | φ₂ | φ₃ | θ₁ | σ² |
|---|---|---|---|---|---|
| Train | 0,664 | −0,095 | 0,157 | −0,569 | 0,0172 |
| Train + Validation | 0,642 | −0,105 | 0,164 | −0,558 | 0,0183 |

- Mọi hệ số có p < 0,001 và ổn định giữa hai lần ước lượng. Thêm một bậc thì hệ số mới không có ý nghĩa (ar.L4 p = 0,95; ma.L2 p = 0,98) và AIC, BIC đều kém hơn, nên (3,1,1) không thiếu bậc. Nghiệm AR có môđun 1,260 và 2,247; nghiệm MA có môđun 1,757: mô hình dừng, khả nghịch.
- Phần dư Train: Ljung–Box (trừ bậc tự do) p = 0,160 ở lag 10, 0,389 ở lag 20. Còn hiệu ứng ARCH mạnh và đuôi dày (kurtosis chưa trừ 3 khoảng 17).

## 5. Kết quả chính

**Test 2026 (N = 175), dự báo một bước, sai số trên giá VND/USD**

| Mô hình | MAE | RMSE | MASE | Relative RMSE | HAC p (bình phương) |
|---|---|---|---|---|---|
| Naive | 18,5629 | 30,0534 | 1,0163 | 1,0000 | – |
| ARIMA(1,1,1) | 18,8044 | 30,0030 | 1,0295 | 0,9983 | 0,892 |
| **ARIMA(3,1,1)** | **18,7363** | **30,4392** | **1,0258** | **1,0128** | **0,469** |
| ARIMA(3,1,4) | 18,6571 | 30,3403 | 1,0214 | 1,0095 | 0,579 |

- Bootstrap khối cho ΔRMSE của (3,1,1) so với Naive: CI 95% [−0,71; 1,43], chứa 0.
- Khoảng 95%: độ phủ 95,43%, độ rộng 139,21 VND, interval score 200,43 (Naive 195,10).
- Đoán đúng chiều trên các phiên có đổi giá: 57,96%. Quy tắc luôn đoán giảm đã đạt 55,41% (87 phiên giảm / 157), nên chênh lệch không có ý nghĩa (p = 0,288). Trên Validation chỉ 45,42%.
- Theo RMSE, (3,1,1) tốt hơn Naive 2,3% trên Train (HAC p = 0,053), 0,3% trên Validation, và kém 1,3% trên Test.
- Sai số Test còn tự tương quan (Ljung–Box(20) p < 0,001) và hiệu ứng ARCH (p = 0,004).
- Dự báo từ mốc cố định 31/12/2025, h = 175: RMSE 133,28 so với 130,41 của Naive; khoảng 95% rộng 1.779 so với 1.260 VND vì ψ(1) ≈ 1,48.
- Các kiểm tra độ vững (ước lượng cuốn chiếu, biến giả ngoại lai, bỏ 105 ngày lễ) không đổi kết luận.

**Bảng 8.4 (dòng ARIMA):** Validation RMSE 43,125272 · Test MAE 18,736338 · Test RMSE 30,439227 · Relative RMSE 1,012838.

## 6. Kết luận và hạn chế

1. USD/VND có cấu trúc ARMA thật trong mẫu: ARIMA(3,1,1) gọn, mọi hệ số có ý nghĩa và ổn định.
2. Cấu trúc này không giúp dự báo ngoài mẫu: trên Validation và Test, không cấu hình ARIMA nào khác Naive có ý nghĩa thống kê.
3. Naive luôn có MAE thấp nhất vì khoảng 10–13% số phiên giá đứng yên.
4. Phần chưa mô hình hóa nằm ở phương sai (ARCH, đuôi dày) và ở chế độ biến động (thấp 2018–2021, cao từ 2022): cơ sở cho GARCH, State Space, SETAR và MS-AR.

**Lưu ý dữ liệu:** nguồn có 569 dòng vi phạm quan hệ OHLC (chỉ dùng Close nên không ảnh hưởng) và 105 ngày lễ bị điền giá phiên trước (đã kiểm tra độ vững ở mục 15.6).
