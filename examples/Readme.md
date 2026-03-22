/**
 * @file  main.c
 * @brief BadCodec サンプル: 汎用C (Linux / macOS / Windows)
 *
 * ヘッダファイル (bad_data.h) とファイル読み込みの両方を示す。
 *
 * ビルド方法:
 *   gcc -std=c99 -O2 main.c bad_decode.cpp -o badplay
 *   ./badplay                        # ヘッダファイルから再生
 *   ./badplay --file output.bad      # ファイルから再生
 *
 * 出力:
 *   ASCII アート (128x64 → 64x32 に縮小) をターミナルに表示
 *   実際の組み込み実装では display_frame() を書き換えること。
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef _WIN32
  #include <windows.h>
  #define SLEEP_MS(ms) Sleep(ms)
#else
  #include <unistd.h>
  #define SLEEP_MS(ms) usleep((ms)*1000)
#endif

#include "bad_decode.h"
#include "bad_data.h"    /* Codec.py -t c で生成。省略も可 */

/* ============================================================
 * 表示: ASCII アート出力 (ターミナル確認用)
 * 組み込みの場合はここを実際のディスプレイ描画に変更する
 * ============================================================ */
static void display_frame(const uint8_t *gram, uint16_t w, uint16_t h)
{
    /* カーソルを先頭に戻す */
    printf("\033[H");

    /* 2ピクセルを1文字で表現 (縦方向2倍に圧縮) */
    for (uint16_t y = 0; y < h; y += 2) {
        for (uint16_t x = 0; x < w; x++) {
            uint16_t bi0 = (uint16_t)(y * w + x);
            uint8_t  p0  = (gram[bi0 >> 3] >> (bi0 & 7)) & 1;
            uint16_t bi1 = (uint16_t)((y+1) * w + x);
            uint8_t  p1  = (y+1 < h)
                         ? ((gram[bi1 >> 3] >> (bi1 & 7)) & 1)
                         : 0;
            /* Unicode ブロック文字: 上半/下半/両方/なし */
            if      (p0 && p1) printf("\xe2\x96\x88"); /* FULL BLOCK */
            else if (p0)       printf("\xe2\x96\x80"); /* UPPER HALF */
            else if (p1)       printf("\xe2\x96\x84"); /* LOWER HALF */
            else               printf(" ");
        }
        printf("\n");
    }
    fflush(stdout);
}

/* ============================================================
 * read コールバック: ヘッダファイル (ROM 配列) から読む
 * ============================================================ */
static uint16_t header_read(bad_addr_t offset, uint8_t *buf, uint16_t len)
{
    memcpy(buf, bad_data + offset, len);
    return len;
}

/* ============================================================
 * read コールバック: FILE* から読む
 * ============================================================ */
static FILE *fp_global = NULL;

static uint16_t file_read(bad_addr_t offset, uint8_t *buf, uint16_t len)
{
    if (!fp_global) return 0;
    fseek(fp_global, (long)offset, SEEK_SET);
    return (uint16_t)fread(buf, 1, len, fp_global);
}

/* ============================================================
 * main
 * ============================================================ */
int main(int argc, char *argv[])
{
    bad_read_fn  read_fn   = header_read;
    const char  *filename  = NULL;

    /* 引数解析 */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--file") == 0 && i+1 < argc) {
            filename = argv[++i];
        }
    }

    /* ファイルモード */
    if (filename) {
        fp_global = fopen(filename, "rb");
        if (!fp_global) {
            fprintf(stderr, "Error: cannot open %s\n", filename);
            return 1;
        }
        read_fn = file_read;
        printf("BadCodec: reading from %s\n", filename);
    } else {
        printf("BadCodec: reading from embedded bad_data[]\n");
    }

    /* デコーダ初期化 */
    static uint8_t   gram[BAD_GRAM_SIZE(128, 64)];
    static uint8_t   prev[BAD_GRAM_SIZE(128, 64)];
    static bad_ctx_t ctx;

    ctx.read     = read_fn;
    ctx.gram     = gram;
    ctx.prev     = prev;
    ctx.buf_size = sizeof(gram);

    if (bad_init(&ctx) != BAD_OK) {
        fprintf(stderr, "bad_init failed\n");
        return 1;
    }

    printf("BadCodec v0.6.0 / Protocol %d\n", BAD_PROTOCOL_VERSION);
    printf("Image: %ux%u  Frames: %u\n",
           ctx.width, ctx.height, ctx.total_frames);

    /* 画面クリア */
    printf("\033[2J\033[H");

    /* 再生ループ */
    for (;;) {
        bad_result_t r = bad_next_frame(&ctx);
        if (r == BAD_OK || r == BAD_EOF) {
            display_frame(ctx.gram, ctx.width, ctx.height);
        }
        if (r == BAD_EOF) {
            bad_rewind(&ctx);
        }

        SLEEP_MS(33);   /* 約30fps */
    }

    if (fp_global) fclose(fp_global);
    return 0;
}
