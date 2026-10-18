# ChatLLM-MOSS.cpp

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-brightgreen.svg)]()
[![Backend](https://img.shields.io/badge/backend-ggml-orange.svg)](https://github.com/ggml-org/ggml)
[![Model](https://img.shields.io/badge/model-MOSS--TTS-blueviolet.svg)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M)

**ChatLLM-MOSS.cpp** là bản fork chuyên biệt từ [chatllm.cpp](https://github.com/foldl/chatllm.cpp) với mục tiêu **tích hợp trọn vẹn kiến trúc MOSS-TTS (MOSS-TTS-Nano-100M, v1.5, v1.0)** thành mô hình C++ native, chạy siêu nhẹ, siêu nhanh trên CPU và GPU dựa trên nền tảng `ggml`.

---

## ✨ Điểm nổi bật (Key Features)

- 🚀 **100% Native C++ Inference**: Toàn bộ pipeline từ Text Prefill, Time $\times$ Depth Autoregressive Generation đến Codec Decoding đều được viết bằng C++ thuần túy, không phụ thuộc Python khi inference.
- 🎙️ **Tích hợp MOSS-Audio-Tokenizer**: Triển khai trực tiếp Neural Audio Codec (1D ConvNeXt stages, Residual LFQ, Transposed Convolutions) trên `ggml`, giải mã âm thanh 48 kHz Stereo độ trung thực cao.
- 🧬 **Zero-shot Voice Cloning**: Nhân bản giọng nói chỉ cần 3 - 6 giây file âm thanh tham chiếu (`.wav`).
- ⚡ **Tốc độ vượt trội (Faster than Real-Time)**:
  - Tốc độ sinh: **~38 - 48 ms / frame** (đạt **21 - 26 fps**).
  - Tỷ lệ thời gian thực (**RTF**): **0.70 – 0.85** trên CPU (thời gian sinh âm thanh ngắn hơn đáng kể so với độ dài giọng nói phát ra).
- 📦 **Tự chứa hoàn toàn (Self-Contained)**:
  - Tích hợp sẵn bộ nạp PCM WAV độc lập (không bắt buộc phải cài `ffmpeg`).
  - Tự động xuất trực tiếp file audio chuẩn RIFF WAV 48.000 Hz Stereo.
- 🎛️ **Đầy đủ tham số tinh chỉnh**: Hỗ trợ tinh chỉnh nhiệt độ lấy mẫu (`text_temp`, `audio_temp`), Top-P, Top-K, Repetition Penalty, âm thanh tham chiếu (`ref_audio`).

---

## 📊 Benchmark hiệu năng (MOSS-TTS-Nano-100M trên CPU)

> *Thử nghiệm đo đạc thực tế trên CPU (đã trừ thời gian load model ~0.7 – 1.0s):*

| Tác vụ | Độ dài âm thanh | Thời gian sinh thuần túy | Tốc độ frame | RTF | Trạng thái |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TTS Tiếng Anh** | 12.56 s (157 frames) | **10.27 s** | 48.90 ms/frame (20.5 fps) | **0.818** | ⚡ Nhanh hơn Real-time |
| **TTS Tiếng Trung** | 15.44 s (193 frames) | **12.76 s** | 46.86 ms/frame (21.3 fps) | **0.826** | ⚡ Nhanh hơn Real-time |
| **Voice Cloning (Short)** | 5.92 s (74 frames) | **4.15 s** | 38.52 ms/frame (26.0 fps) | **0.701** | ⚡ Nhanh hơn Real-time |
| **Voice Cloning (Long)** | 14.96 s (187 frames) | **13.16 s** | 47.04 ms/frame (21.3 fps) | **0.879** | ⚡ Nhanh hơn Real-time |

*Đặc thù kiến trúc MOSS-TTS: 1 frame latent tương đương 80 ms âm thanh (12.5 Hz). Tốc độ sinh 38 - 48 ms/frame đồng nghĩa với việc mô hình sinh nhanh gấp 1.7 – 2.1 lần tốc độ nói tự nhiên.*

---

## 🛠️ Hướng dẫn cài đặt & Biên dịch (Build)

### 1. Yêu cầu hệ thống
- CMake $\ge$ 3.20
- Trình biên dịch C++20 (MSVC trên Windows, GCC 11+ hoặc Clang 14+ trên Linux/macOS)

### 2. Biên dịch
```bash
# Clone repository
git clone https://github.com/DDATT/chatllm-moss.cpp.git
cd chatllm-moss.cpp

# Tạo thư mục build và cấu hình
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Biên dịch chương trình chính
cmake --build build --config Release --target main -j
```
File thực thi sẽ nằm tại `build/bin/Release/main.exe` (Windows) hoặc `build/bin/main` (Linux).

---

## 🔄 Chuyển đổi mô hình (Model Conversion)

Để chuyển đổi trọng số từ HuggingFace sang định dạng nhị phân `.bin` của `chatllm.cpp`:

### 1. Chuẩn bị môi trường Python
```bash
pip install torch transformers huggingface_hub
```

### 2. Tải trọng số MOSS-TTS-Nano-100M
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('OpenMOSS-Team/MOSS-TTS-Nano-100M', local_dir='./MOSS-TTS-Nano-100M')"
```

### 3. Chuyển đổi sang file `.bin`
```bash
python convert.py -i ./MOSS-TTS-Nano-100M -o moss-tts-nano.bin -t f32
```
*Lưu ý: Script sẽ tự động nhận diện kiến trúc MOSS-TTS và nhúng toàn bộ trọng số của Audio Tokenizer Codec vào chung một file nhị phân duy nhất (~345 MB).*

---

## 🚀 Hướng dẫn sử dụng (Usage)

### 1. Tổng hợp giọng nói cơ bản (Text-to-Speech)
```bash
./build/bin/Release/main.exe -m moss-tts-nano.bin -p "Artificial intelligence and neural speech synthesis have made tremendous progress in recent years." --tts_export output.wav
```

### 2. Nhân bản giọng nói (Zero-shot Voice Cloning)
Chỉ định file audio tham chiếu (độ dài lý tưởng 3 - 6 giây) qua tham số `--set ref_audio <file.wav>`:
```bash
./build/bin/Release/main.exe -m moss-tts-nano.bin -p "Hello! This voice has been cloned locally using MOSS TTS on chatllm-moss.cpp." --set ref_audio reference.wav --tts_export cloned_output.wav
```

### 3. Tinh chỉnh các tham số lấy mẫu (Sampling Controls)
Bạn có thể điều chỉnh độ biểu cảm hoặc tốc độ thông qua cờ `--set`:
```bash
./build/bin/Release/main.exe -m moss-tts-nano.bin \
  -p "Hello, world!" \
  --set text_temp 1.2 \
  --set audio_temp 1.5 \
  --set text_top_p 0.95 \
  --set audio_top_p 0.8 \
  --set audio_rep_penalty 1.05 \
  --tts_export output.wav
```

---

## 📁 Cấu trúc mã nguồn tích hợp

- [`models/moss_tts.h`](models/moss_tts.h): Khai báo kiến trúc MOSS-TTS và MOSS-Audio-Tokenizer (Codec, Residual LFQ, Multi-stage Transposed Convolutions, Local & Global backbone config).
- [`models/moss_tts.cpp`](models/moss_tts.cpp): Hiện thực hóa toàn bộ pipeline xử lý âm thanh, Time $\times$ Depth autoregressive generation loop, benchmark instrumentation và bộ nạp WAV độc lập.
- [`convert.py`](convert.py): Module trích xuất và đóng gói tensor từ HuggingFace safetensors sang định dạng ggml tensor.
- [`src/main.cpp`](src/main.cpp): Hỗ trợ xuất trực tiếp chuẩn RIFF WAV header và hiển thị đo đạc thời gian nạp / thực thi.

---

## 📜 Giấy phép (License)
Dự án được phân phối dưới giấy phép **MIT License**. Vui lòng tham khảo file [LICENSE](LICENSE) để biết thêm chi tiết. Trọng số mô hình MOSS-TTS tuân thủ theo giấy phép của [OpenMOSS Team](https://huggingface.co/OpenMOSS-Team).
