/**
 * @file  main.c
 * @brief BadCodecPlayer - ESP32-C3 Mini / SSD1306 72x40
 * @version v0.6.0  (Protocol 514, SPEC rev.19)
 *
 * ESP32-C3 Mini + SSD1306 128x64 OLED (I2C)
 * BadCodec v0.6.0 フォーマットの動画を Flash から再生する。
 *
 * 動作:
 *   1. bad_data.h の埋め込み配列から BadCodec ストリームを読む
 *   2. bad_decode でフレームをデコード (gram: 72x40 1bit/px)
 *   3. ssd1306_draw_gram で SSD1306 framebuffer に転送
 *   4. ssd1306_flush で OLED に表示
 *   5. EOF で先頭に戻りループ再生
 *
 * 設定:
 *   include/bad_player_config.h で GPIO・解像度・FPSを変更
 *
 * ファイルの準備:
 *   python3 tools/Codec.py -t e -p ./frames -n frame_ -s 0001 -e XXXX -o out.bad
 *   python3 tools/Codec.py -t c -i out.bad -H bad_data.h
 *   → 生成した bad_data.h を include/ に配置
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "bad_player_config.h"
#include "ssd1306.h"
#include "bad_decode.h"

/* ============================================================
 * bad_data.h の選択
 * BAD_SOURCE_FLASH が定義されている場合のみ include
 * ============================================================ */
#ifdef BAD_SOURCE_FLASH
  #include "bad_data.h"   /* Codec.py -t c で生成 → include/ に配置 */
#endif

static const char *TAG = "BadCodec";

/* ============================================================
 * gram / prev バッファ
 * 72x40 / 8 = 360 bytes 各2面 = 720 bytes
 * ============================================================ */
static uint8_t   gram[BAD_GRAM_SIZE(BAD_VIDEO_WIDTH, BAD_VIDEO_HEIGHT)];
static uint8_t   prev[BAD_GRAM_SIZE(BAD_VIDEO_WIDTH, BAD_VIDEO_HEIGHT)];
static bad_ctx_t ctx;

/* ============================================================
 * read コールバック: Flash (XIP) から直接コピー
 * bad_data[] は DRAM / Flash キャッシュ経由でアクセス可能
 * ============================================================ */
#ifdef BAD_SOURCE_FLASH
static uint16_t flash_read(bad_addr_t offset, uint8_t *buf, uint16_t len)
{
    memcpy(buf, bad_data + offset, len);
    return len;
}
#endif

/* ============================================================
 * read コールバック: SD カード (FAT) から読む
 * BAD_SOURCE_SD 定義時に使用
 * ============================================================ */
#ifdef BAD_SOURCE_SD
#include "esp_vfs_fat.h"
#include "driver/sdmmc_host.h"
#include "sdmmc_cmd.h"
#include <stdio.h>
static FILE  *sd_fp    = NULL;
static long   sd_last  = -1;

static uint16_t sd_read(bad_addr_t offset, uint8_t *buf, uint16_t len)
{
    if (!sd_fp) return 0;
    if ((long)offset != sd_last) {
        fseek(sd_fp, (long)offset, SEEK_SET);
    }
    size_t n = fread(buf, 1, len, sd_fp);
    sd_last = (long)offset + (long)n;
    return (uint16_t)n;
}

static bool sd_init(void)
{
    esp_vfs_fat_sdmmc_mount_config_t mcfg = {
        .format_if_mount_failed = false,
        .max_files              = 4,
        .allocation_unit_size   = 16 * 1024,
    };
    sdmmc_card_t *card;
    sdmmc_host_t  host = SDMMC_HOST_DEFAULT();
    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width = 1;
    if (esp_vfs_fat_sdmmc_mount("/sdcard", &host, &slot, &mcfg, &card) != ESP_OK) {
        ESP_LOGE(TAG, "SD mount failed");
        return false;
    }
    sd_fp = fopen(BAD_SD_FILE, "rb");
    if (!sd_fp) {
        ESP_LOGE(TAG, "Cannot open %s", BAD_SD_FILE);
        return false;
    }
    ESP_LOGI(TAG, "SD OK: %s", BAD_SD_FILE);
    return true;
}
#endif /* BAD_SOURCE_SD */

/* ============================================================
 * 再生タスク
 * ============================================================ */
static void player_task(void *arg)
{
    /* BadCodec 初期化 */
    ctx.gram     = gram;
    ctx.prev     = prev;
    ctx.buf_size = sizeof(gram);

#ifdef BAD_SOURCE_FLASH
    ctx.read = flash_read;
    ESP_LOGI(TAG, "Source: Flash (bad_data[])");
#endif

#ifdef BAD_SOURCE_SD
    if (!sd_init()) { vTaskDelete(NULL); return; }
    ctx.read = sd_read;
    ESP_LOGI(TAG, "Source: SD %s", BAD_SD_FILE);
#endif

    if (bad_init(&ctx) != BAD_OK) {
        ESP_LOGE(TAG, "bad_init failed");
        /* エラー画面: 画面全体を点滅 */
        for (;;) {
            memset(ssd1306_fb, 0xFF, sizeof(ssd1306_fb));
            ssd1306_flush();
            vTaskDelay(pdMS_TO_TICKS(500));
            ssd1306_clear();
            ssd1306_flush();
            vTaskDelay(pdMS_TO_TICKS(500));
        }
    }

    ESP_LOGI(TAG, "Playing %ux%u  %u frames  %u fps",
             ctx.width, ctx.height, ctx.total_frames,
             (unsigned)(1000 / BAD_FRAME_MS));

    /* 再生ループ */
    for (;;) {
        TickType_t t0 = xTaskGetTickCount();

        bad_result_t r = bad_next_frame(&ctx);

        if (r == BAD_OK || r == BAD_EOF) {
            /* gram → SSD1306 framebuffer → 転送 */
            ssd1306_clear();
            ssd1306_draw_gram(ctx.gram,
                              ctx.width, ctx.height,
                              BAD_X_OFFSET, BAD_Y_OFFSET);
            ssd1306_flush();
        }

        if (r == BAD_EOF) {
            ESP_LOGI(TAG, "Loop");
            bad_rewind(&ctx);
#ifdef BAD_SOURCE_SD
            sd_last = -1;
#endif
        }

        /* フレームレート制御 */
        vTaskDelayUntil(&t0, pdMS_TO_TICKS(BAD_FRAME_MS));
    }
}

/* ============================================================
 * app_main
 * ============================================================ */
void app_main(void)
{
    ESP_LOGI(TAG, "BadCodecPlayer v0.6.0  Protocol 514  SPEC rev.19");

    /* SSD1306 初期化 */
    ssd1306_init();

    /* 起動画面: 全画素点灯 → クリア */
    memset(ssd1306_fb, 0xFF, sizeof(ssd1306_fb));
    ssd1306_flush();
    vTaskDelay(pdMS_TO_TICKS(300));
    ssd1306_clear();
    ssd1306_flush();

    /* 再生タスク起動 (スタック 6KB, 優先度 5) */
    xTaskCreate(player_task, "player", 6144, NULL, 5, NULL);
}
