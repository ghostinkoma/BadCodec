# BadCodecPlayer

A video + audio player for ESP32-C3 Super Mini + SSD1306 128×64 OLED.  
Plays BadCodec-encoded video and IMA-ADPCM audio simultaneously.

---

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| MCU | ESP32-C3 Super Mini |
| Display | SSD1306 128×64 OLED (I2C) |
| Audio output | GPIO10 → RC LPF → amplifier → speaker |

---

## Wiring

### OLED (I2C)

```
ESP32-C3          SSD1306
GPIO8  (SDA) ──── SDA
GPIO9  (SCL) ──── SCL
3.3V         ──── VCC
GND          ──── GND
```

> **Note:** GPIO8 and GPIO9 are strapping pins on the ESP32-C3, but they are
> safe to use for I2C as long as **no external pull-up resistors** are fitted.
> Adding external pull-ups (e.g. 3.3 kΩ) will cause boot failure.

### Audio Output (LEDC PWM)
-You can change audio out pin in config.h

Single-stage LPF (minimum):
```
GPIO10 → ── 1μF to 33μF ── Speaker

```

Two-stage LPF (recommended for better quality):
```
GPIO10 → R=10kΩ ─┬─ C=100nF ─ GND
                  └─ R=10kΩ ─┬─ C=100nF ─ GND
                              └─ C=10μF  ─ Amp input
```

> **⚠️ Important — LPF component values**  
> The PWM carrier frequency is 39 kHz. The old ΣΔ-era values
> (R=1 kΩ, C=10 nF, fc ≈ 16 kHz) will let the carrier pass through and
> cause audible noise. **You must change to R=10 kΩ, C=100 nF.**

---

## Audio Output Method

Since v2.0.0, the driver uses the **ESP32-C3 LEDC peripheral (hardware PWM)**
instead of the previous software sigma-delta modulation.

| Parameter | Value |
|-----------|-------|
| PWM carrier frequency | 39,062 Hz (80 MHz ÷ 2¹¹) |
| PWM resolution | 11 bit (0–2047) |
| Dynamic range | ≈ 66 dB |
| Sample rate | 16,000 Hz (read from WAV header) |
| ISR CPU usage | < 0.1% |

The carrier is generated entirely by hardware; the ISR only updates the duty
cycle at ≈16 kHz. This eliminates the scheduling conflicts that caused noise
with the software ΣΔ approach.

---

## Build Environment

- **PlatformIO** with the ESP-IDF framework
- Board: `esp32-c3-devkitm-1`

### platformio.ini (reference)

```ini
[env:esp32c3-supermini]
platform  = espressif32
board     = esp32-c3-devkitm-1
framework = espidf
board_build.partitions = partitions.csv
monitor_speed  = 115200
monitor_filter = esp32_exception_decoder
upload_speed   = 921600

build_flags =
    -DCONFIG_FREERTOS_HZ=1000
```

---

## Project Structure

```
c3SuperMini/
├── src/
│   ├── main.c           Application entry point (task management)
│   ├── adpcm_drv.c      IMA-ADPCM decoder + LEDC PWM audio output
│   ├── ssd1306_drv.c    SSD1306 I2C display driver
│   └── bad_decode.c     BadCodec v0.5.1 video decoder
├── include/
│   ├── config.h         Global settings (pins, timing, debug switches)
│   ├── adpcm_drv.h      Audio driver API
│   ├── ssd1306_drv.h    Display driver API
│   ├── bad_decode.h     Video decoder API
│   ├── bad_data.h       BadCodec video data (C array)
│   └── adpcm4.h         IMA-ADPCM audio data (C array)
├── partitions.csv       Custom partition table
└── platformio.ini       PlatformIO configuration
```

---

## Preparing Data Files

### Video data (bad_data.h)

Encode your video with the BadCodec encoder to produce a `.bad` file,
then convert it to a C header:

```bash
python3 bad2header.py -i output.bad -o include/bad_data.h
```

The header must define:

```c
#define BAD_DATA_SIZE   /* total byte count */
const uint8_t bad_data[];
```

### Audio data (adpcm4.h)

Use the bundled `Wave2adpcmH.py` script to convert a WAV file:

```bash
# Convert PCM WAV to IMA-ADPCM header (requires ffmpeg)
python3 Wave2adpcmH.py -i audio.wav -o include/adpcm4.h

# Specify sample rate explicitly
python3 Wave2adpcmH.py -i audio.wav -o include/adpcm4.h -r 16000
```

After conversion, set `CFG_AUDIO_SR` in `config.h` to match the sample
rate reported by the script.

The generated header defines:

```c
#define BAD_AUDIO_SIZE         /* total byte count */
#define BAD_AUDIO_SAMPLE_RATE  /* sample rate in Hz */
#define BAD_AUDIO_CHANNELS     /* number of channels */
#define BAD_AUDIO_BLOCK_ALIGN  /* block alignment */
const uint8_t bad_audio_data[];
```

Installing ffmpeg:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

---

## Configuration (config.h)

```c
/* I2C */
#define CFG_I2C_SDA     8       /* SDA GPIO number */
#define CFG_I2C_SCL     9       /* SCL GPIO number */
#define CFG_I2C_FREQ_HZ 400000  /* 400 kHz */

/* OLED */
#define CFG_OLED_ADDR   0x3D    /* I2C address (change to 0x3C if needed) */

/* Audio output */
#define CFG_AUDIO_PIN_P 10      /* PWM output GPIO */
#define CFG_AUDIO_SR    16000U  /* Sample rate — must match the WAV file */
#define CFG_AUDIO_VOL   120U    /* Volume: 0 = mute, 256 = full scale */

/* Frame timing */
#define CFG_FRAME_MS    29U     /* Frame period in ms (≈ 34.5 fps) */

/* Debug switches */
#define CFG_VIDEO_ENABLE  1     /* 1 = video on,  0 = video off */
#define CFG_AUDIO_ENABLE  1     /* 1 = audio on,  0 = audio off */
```

### Finding the OLED I2C address

The SSD1306 uses either `0x3C` or `0x3D` depending on the module.
Run an I2C scanner sketch to identify the correct address if unknown.

### Debug / Isolation procedure

If playback does not work, isolate the issue step by step:

```
Step 1: CFG_VIDEO_ENABLE=1, CFG_AUDIO_ENABLE=0  →  verify video only
Step 2: CFG_VIDEO_ENABLE=0, CFG_AUDIO_ENABLE=1  →  verify audio only (check serial log)
Step 3: CFG_VIDEO_ENABLE=1, CFG_AUDIO_ENABLE=1  →  both together (normal operation)
```

---

## Signal Path

```
Flash (adpcm4.h)
  └─ adpcm_task  [IMA-ADPCM decode, priority 4]
       └─ ring buffer  [RAM, 65,536 samples = 4.1 s]
            └─ gptimer ISR  [≈ 16 kHz]
                 └─ LEDC PWM duty update  [hardware]
                      └─ GPIO10 → LPF → amplifier → speaker

Flash (bad_data.h)
  └─ player_task  [BadCodec decode + display, priority 5]
       ├─ bad_next_frame() → ssd1306_blit_gram() → ssd1306_flush()
       └─ xSemaphoreTake  [waits for frame-sync from gptimer ISR]
```

### Task priorities

| Task | Priority | Role |
|------|----------|------|
| gptimer ISR | highest (interrupt) | PWM duty update + frame-sync semaphore |
| player_task | 5 | video decode and display |
| adpcm_task  | 4 | audio decode and ring buffer fill |

`adpcm_task` runs at a lower priority than `player_task` to prevent it
from preempting the I2C transfer inside `ssd1306_flush()`.
The 4.1-second ring buffer provides ample headroom so that brief periods
where `adpcm_task` cannot run do not cause audio glitches.

---

## Memory Usage (reference)

```
RAM:   ≈ 4.5%  (ring buffer 128 KB + static variables ≈ 15 KB)
Flash: ≈ 71%   (includes video and audio data arrays)
```

Total RAM on ESP32-C3 is 320 KB.  
The ring buffer size (`ADPCM_RING_SIZE` in `adpcm_drv.h`) can be adjusted,
but values above 131,072 (256 KB) will overflow the DRAM region.

---

## License

- `bad_decode.h` / `bad_decode.c`: Non-Commercial Use Only (ghostinkoma@gmail.com)
- All other files: see the project author's license
