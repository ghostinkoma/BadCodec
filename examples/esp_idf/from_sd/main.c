/**
 * @file  main.c
 * @brief BadCodec サンプル: ESP-IDF / SD カードから再生
 * @version v0.6.0  (Protocol 514, SPEC rev.19)
 *
 * SD カード: SDMMC (1-line) または SPI 接続
 * ディスプレイ: SSD1306 128x64 (esp-idf-lib)
 *
 * SD カードのルートに output.bad を置いてください。
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "driver/sdmmc_host.h"
#include "sdmmc_cmd.h"
#include "bad_decode.h"

static const char *TAG = "BadCodec";

/* ============================================================
 * SD カード設定 (1-line SDMMC; SPI の場合は変更すること)
 * ============================================================ */
#define MOUNT_POINT  "/sdcard"
#define VIDEO_FILE   MOUNT_POINT "/output.bad"

static void sd_init(void)
{
    esp_vfs_fat_sdmmc_mount_config_t mount_cfg = {
        .format_if_mount_failed = false,
        .max_files              = 4,
        .allocation_unit_size   = 16 * 1024,
    };
    sdmmc_card_t *card;
    sdmmc_host_t  host = SDMMC_HOST_DEFAULT();
    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width = 1;

    ESP_ERROR_CHECK(esp_vfs_fat_sdmmc_mount(
        MOUNT_POINT, &host, &slot, &mount_cfg, &card));
    ESP_LOGI(TAG, "SD mounted: %s %lluMB",
             card->cid.name,
             (uint64_t)card->csd.capacity * card->csd.sector_size / (1024*1024));
}

/* ============================================================
 * ディスプレイ (display_frame は from_flash と同一のため省略)
 * 実際のコードではここに SSD1306 初期化・描画コードを置く
 * ============================================================ */
static void display_init(void) { /* TODO */ }
static void display_frame(const uint8_t *g, uint16_t w, uint16_t h)
{
    (void)g; (void)w; (void)h; /* TODO: 実際のディスプレイ描画 */
    ESP_LOGD(TAG, "frame %ux%u", w, h);
}

/* ============================================================
 * BadCodec デコーダ
 * ============================================================ */
static uint8_t   gram[BAD_GRAM_SIZE(128, 64)];
static uint8_t   prev_buf[BAD_GRAM_SIZE(128, 64)];
static bad_ctx_t ctx;
static FILE     *sd_fp    = NULL;
static long      last_pos = -1;

/* read コールバック: fseek + fread */
static uint16_t sd_read(bad_addr_t offset, uint8_t *buf, uint16_t len)
{
    if (!sd_fp) return 0;
    if ((long)offset != last_pos) {
        fseek(sd_fp, (long)offset, SEEK_SET);
    }
    size_t n = fread(buf, 1, len, sd_fp);
    last_pos = (long)offset + (long)n;
    return (uint16_t)n;
}

/* ============================================================
 * メインタスク
 * ============================================================ */
static void video_task(void *arg)
{
    sd_init();
    display_init();

    sd_fp = fopen(VIDEO_FILE, "rb");
    if (!sd_fp) {
        ESP_LOGE(TAG, "Cannot open %s", VIDEO_FILE);
        vTaskDelete(NULL);
    }

    ctx.read     = sd_read;
    ctx.gram     = gram;
    ctx.prev     = prev_buf;
    ctx.buf_size = sizeof(gram);

    if (bad_init(&ctx) != BAD_OK) {
        ESP_LOGE(TAG, "bad_init failed");
        fclose(sd_fp);
        vTaskDelete(NULL);
    }

    ESP_LOGI(TAG, "Playing %ux%u %u frames",
             ctx.width, ctx.height, ctx.total_frames);

    for (;;) {
        TickType_t t0 = xTaskGetTickCount();

        bad_result_t r = bad_next_frame(&ctx);
        if (r == BAD_OK || r == BAD_EOF) {
            display_frame(ctx.gram, ctx.width, ctx.height);
        }
        if (r == BAD_EOF) {
            bad_rewind(&ctx);
            last_pos = -1;
        }

        vTaskDelayUntil(&t0, pdMS_TO_TICKS(33));
    }
}

void app_main(void)
{
    xTaskCreate(video_task, "video", 8192, NULL, 5, NULL);
}
