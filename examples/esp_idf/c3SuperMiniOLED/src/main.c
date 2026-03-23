/**
 * @file  main.c
 * @brief BadCodecPlayer - ESP32-C3 Super Mini / SSD1306 72x40 builtin
 * @version v0.6.0  (Protocol 514, SPEC rev.19)
 *
 * Reads bad_data.h from Flash, decodes with BadCodec v0.6.0,
 * blits 72x40 gram to SSD1306 at (CFG_X_OFFSET=27, CFG_Y_OFFSET=12).
 * Loops forever.
 *
 * To use SD card instead of Flash:
 *   Edit include/config.h: comment CFG_SOURCE_FLASH, enable CFG_SOURCE_SD
 */

 #include <string.h>
 #include "freertos/FreeRTOS.h"
 #include "freertos/task.h"
 #include "esp_log.h"
 
 #include "config.h"
 #include "ssd1306_drv.h"
 #include "bad_decode.h"
 #include "bad_data.h"
 
 static const char *TAG = "BadCodec";
 
 /* ---- decoder buffers: 72*40/8 = 360 bytes each ---- */
 static uint8_t   s_gram[BAD_GRAM_SIZE(CFG_VIDEO_W, CFG_VIDEO_H)];
 static uint8_t   s_prev[BAD_GRAM_SIZE(CFG_VIDEO_W, CFG_VIDEO_H)];
 static bad_ctx_t s_ctx;
 
 /* ---- read callbacks ---- */
 
 #ifdef CFG_SOURCE_FLASH
 static uint16_t flash_read(bad_addr_t off, uint8_t *buf, uint16_t len)
 {
     memcpy(buf, bad_data + off, len);
     return len;
 }
 #endif
 
 #ifdef CFG_SOURCE_SD
 #include "esp_vfs_fat.h"
 #include "driver/sdmmc_host.h"
 #include "sdmmc_cmd.h"
 #include <stdio.h>
 static FILE *s_sd_fp   = NULL;
 static long  s_sd_last = -1L;
 static uint16_t sd_read(bad_addr_t off, uint8_t *buf, uint16_t len)
 {
     if (!s_sd_fp) return 0;
     if ((long)off != s_sd_last) fseek(s_sd_fp, (long)off, SEEK_SET);
     size_t n = fread(buf, 1, len, s_sd_fp);
     s_sd_last = (long)off + (long)n;
     return (uint16_t)n;
 }
 static bool sd_mount(void)
 {
     esp_vfs_fat_sdmmc_mount_config_t mc = {
         .format_if_mount_failed = false,
         .max_files = 2,
         .allocation_unit_size = 16 * 1024,
     };
     sdmmc_card_t *card;
     sdmmc_host_t  host = SDMMC_HOST_DEFAULT();
     sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
     slot.width = 1;
     if (esp_vfs_fat_sdmmc_mount("/sdcard", &host, &slot, &mc, &card) != ESP_OK)
         return false;
     s_sd_fp = fopen(CFG_SD_FILE, "rb");
     return s_sd_fp != NULL;
 }
 #endif
 
 /* ---- error indicator: blink full screen ---- */
 static void show_error(void)
 {
     for (;;) {
         memset(g_fb, 0xFF, sizeof(g_fb));
         ssd1306_flush();
         vTaskDelay(pdMS_TO_TICKS(400));
         ssd1306_clear();
         ssd1306_flush();
         vTaskDelay(pdMS_TO_TICKS(400));
     }
 }
 
 /* ---- player task ---- */
 static void player_task(void *arg)
 {
     (void)arg;
 
 #ifdef CFG_SOURCE_FLASH
     s_ctx.read = flash_read;
     ESP_LOGI(TAG, "Source: Flash (bad_data[])");
 #endif
 
 #ifdef CFG_SOURCE_SD
     if (!sd_mount()) {
         ESP_LOGE(TAG, "SD mount failed");
         show_error();
     }
     s_ctx.read = sd_read;
     ESP_LOGI(TAG, "Source: SD %s", CFG_SD_FILE);
 #endif
 
     s_ctx.gram     = s_gram;
     s_ctx.prev     = s_prev;
     s_ctx.buf_size = (uint16_t)sizeof(s_gram);
 
     if (bad_init(&s_ctx) != BAD_OK) {
         ESP_LOGE(TAG, "bad_init failed");
         show_error();
     }
 
     ESP_LOGI(TAG, "%ux%u  %u frames  ~%ums/frame",
              s_ctx.width, s_ctx.height,
              s_ctx.total_frames, (unsigned)CFG_FRAME_MS);
 
     for (;;) {
         TickType_t t0 = xTaskGetTickCount();
 
         bad_result_t r = bad_next_frame(&s_ctx);
 
         if (r == BAD_OK || r == BAD_EOF) {
             ssd1306_clear();
             ssd1306_blit_gram(s_ctx.gram,
                               (int)s_ctx.width,
                               (int)s_ctx.height);
             ssd1306_flush();
         }
 
         if (r == BAD_EOF) {
             bad_rewind(&s_ctx);
 #ifdef CFG_SOURCE_SD
             s_sd_last = -1L;
 #endif
         } else if (r != BAD_OK) {
             /* デコードエラー: ストリームが壊れている可能性があるため
              * 先頭に戻ってループ再生を継続する。               */
             ESP_LOGW(TAG, "frame %u: decode error %d, rewinding",
                      (unsigned)s_ctx.current_frame, (int)r);
             bad_rewind(&s_ctx);
 #ifdef CFG_SOURCE_SD
             s_sd_last = -1L;
 #endif
         }
 
         vTaskDelayUntil(&t0, pdMS_TO_TICKS(CFG_FRAME_MS));
     }
 }
 
 /* ---- app_main ---- */
 void app_main(void)
 {
     ESP_LOGI(TAG, "BadCodecPlayer v0.6.0  Protocol 514");
 
     ssd1306_init();
 
     /* startup flash */
     memset(g_fb, 0xFF, sizeof(g_fb));
     ssd1306_flush();
     vTaskDelay(pdMS_TO_TICKS(200));
     ssd1306_clear();
     ssd1306_flush();
 
     xTaskCreate(player_task, "player", 6144, NULL, 5, NULL);
 }
 
