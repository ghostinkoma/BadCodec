/**
 * @file  BadCodecPlayer.cpp
 * @brief BadCodec v0.5.5 Arduino ライブラリ 実装
 */

#include "BadCodecPlayer.h"

#ifdef ARDUINO
  #include <Arduino.h>
#endif

BadCodecPlayer::BadCodecPlayer()
    : _display(nullptr)
{
    /* コンテキストをゼロ初期化 */
    uint8_t *p = (uint8_t*)&_ctx;
    for (uint16_t i = 0; i < sizeof(_ctx); i++) p[i] = 0;
}

bool BadCodecPlayer::begin(bad_read_fn    read_fn,
                           bad_display_fn display_fn,
                           uint16_t       width,
                           uint16_t       height)
{
    (void)width; (void)height; /* ヘッダから自動取得 */

    _display       = display_fn;
    _ctx.read      = read_fn;
    _ctx.gram      = _gram;
    _ctx.prev      = _prev;
    _ctx.buf_size  = sizeof(_gram);

    bad_result_t r = bad_init(&_ctx);
    return (r == BAD_OK);
}

bad_result_t BadCodecPlayer::stepOne()
{
    bad_result_t r = bad_next_frame(&_ctx);
    if (r == BAD_OK || r == BAD_EOF) {
        if (_display)
            _display(_ctx.gram, _ctx.width, _ctx.height);
    }
    if (r == BAD_EOF)
        bad_rewind(&_ctx);
    return r;
}

void BadCodecPlayer::play(uint16_t interval_ms)
{
    for (;;) {
        uint32_t t0 = millis();
        stepOne();
        uint32_t elapsed = (uint32_t)(millis() - t0);
        if (elapsed < interval_ms)
            delay(interval_ms - elapsed);
    }
}

void BadCodecPlayer::rewind()
{
    bad_rewind(&_ctx);
}
