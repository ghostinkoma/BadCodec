/**
 * @file  BadCodecPlayer.h
 * @brief BadCodec v0.5.5 Arduino ライブラリ
 *
 * 対応ボード:
 *   LGT8F328P / ATmega328P (Arduino Uno, Nano, Mini 等)
 *   その他 Arduino 互換ボード
 *
 * 依存ライブラリ:
 *   SD.h (SD カード再生時)
 *
 * 使い方:
 *   BadCodecPlayer player;
 *   player.begin(read_fn, display_fn, width, height);
 *   player.play();    // ループ再生
 *   player.stepOne(); // 1フレーム進める
 */

#ifndef BAD_CODEC_PLAYER_H
#define BAD_CODEC_PLAYER_H

#include <stdint.h>
#include "bad_decode.h"

/* ============================================================
 * コールバック型
 * ============================================================ */

/**
 * 表示コールバック
 * @param gram   1bit グレースケールバッファ (width*height/8 bytes)
 * @param width  画像幅 (px)
 * @param height 画像高さ (px)
 */
typedef void (*bad_display_fn)(const uint8_t *gram,
                               uint16_t       width,
                               uint16_t       height);

/* ============================================================
 * BadCodecPlayer クラス
 * ============================================================ */

class BadCodecPlayer {
public:
    BadCodecPlayer();

    /**
     * 初期化
     * @param read_fn    bad_read_fn コールバック
     * @param display_fn 表示コールバック
     * @param width      画像幅  (ヘッダから自動取得も可能)
     * @param height     画像高さ (ヘッダから自動取得も可能)
     * @return true = 成功
     */
    bool begin(bad_read_fn    read_fn,
               bad_display_fn display_fn,
               uint16_t       width  = 0,
               uint16_t       height = 0);

    /** 1フレーム進めて表示する。EOF で先頭に戻る */
    bad_result_t stepOne();

    /** ループ再生 (interval_ms ミリ秒間隔) */
    void play(uint16_t interval_ms = 33);

    /** 先頭フレームに戻る */
    void rewind();

    uint16_t currentFrame() const { return bad_current_frame(&_ctx); }
    uint16_t totalFrames()  const { return bad_total_frames(&_ctx);  }
    uint16_t width()        const { return _ctx.width;               }
    uint16_t height()       const { return _ctx.height;              }

private:
    bad_ctx_t      _ctx;
    bad_display_fn _display;
    uint8_t        _gram[BAD_GRAM_SIZE(128, 64)];
    uint8_t        _prev[BAD_GRAM_SIZE(128, 64)];
};

#endif /* BAD_CODEC_PLAYER_H */
