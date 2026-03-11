# Changelog

All notable changes to BadCodec will be documented in this file.

Version format: `MAJOR.MINOR.PATCH`

- `0.x.x` : Pre-release. Encode/decode integrity not yet fully verified.
- `1.0.0` : First stable release. Full encode/decode test suite passed.

-----

## [Unreleased]

- マイコン向けCデコーダ実装
- エンコード/デコード整合性テストスイート
- 各ターゲット実機検証（LGT8F328 / ESP32-Cx / RP2350）

-----

## [0.5.1] - 2026-03-10

### Changed

- **ヘッダーチェックサムをMD5からFletcher-16に変更**
  - ヘッダーサイズ: 33バイト → 19バイト（14バイト削減）
  - マイコンデコーダのRAM使用量: -14バイト
  - マイコンデコーダのFlash使用量: 数百バイト削減
  - 8ビット加算のみで完結・テーブル不要
- バージョン番号: 500 → 510
- `hashlib` 依存を完全削除

### Spec

- SPEC.md rev.11
- 3-3節: MD5 → Fletcher-16に全面改訂
- C言語実装例追記

-----

## [0.5.0] - 2026-03-10

### Added

- **フレームFOR最適化** (`merge_frame_for`)
  - 連続する同一フレーム命令をFOR命令に自動集約
  - エンコードをPhase1（並列エンコード）/ Phase2（FORマージ）/ Phase3（書き込み）の3フェーズに分離
  - FOR 0（2回）・FOR 1（3回）の生成禁止ルールを正確に実装
- **デコードUI**
  - プログレスバー・フレーム情報・書き込みパス表示
- **マルチCPU対応**（エンコード・デコード両対応）
  - 1フレーム1コアの静的分担
  - 書き出しレギュレーション（最小待ちポインタで即時書き出し・それ以外はメモリ保持）
- **Self-Verify機構**
  - エンコード後即デコードして元データと比較
  - 不一致時はMASTER_FRAMEでフォールバック再エンコード
- **デコードCLI引数** `-n` / `-s` / `-e` 対応
- **拡張命令体系** `0xFF` プレフィックス（DELTA_FRAME予約）

### Changed

- INVERT_PREV_FRAME: 単純論理反転（NOT）に確定・パディングなし
- MASTER_FRAME: リトルエンディアン・行方向に確定（MASTER_BLOCKと同一）
- DELTA_FRAMEを通常命令空間から拡張命令空間（`0xFF 0x01`）へ移動

### Spec

- SPEC.md rev.7〜rev.10

-----

## [0.4.0] - 2026-03-10

### Added

- **ブロック・フレーム最適化アルゴリズム** 完全実装
  - 全候補試算・最小バイト数採用
  - FOR連続圧縮試算（N≧4でFOR使用）
  - フレーム単位最適表現選択フロー
- **CLI仕様** 完全定義（`-t/-p/-n/-s/-e/-o/-i`）
- **エンコードUI** プログレスバー・ブロックマップ・移動平均圧縮率

### Spec

- SPEC.md rev.4

-----

## [0.3.0] - 2026-03-10

### Added

- **エンコード中アナライズUI** ソースコードより仕様抽出
  - ブロック種別カラー定義（S/F/R/X/M/I）
  - 直近90フレーム移動平均圧縮率
  - ビジュアルブロックマップ（最大8行×30列）

### Spec

- SPEC.md rev.5

-----

## [0.2.0] - 2026-03-10

### Fixed

- **命令分木の矛盾解消**
  - RLE_BLOCK: bit5=1配下の競合を解決（bit4=0でRLE_BLOCK識別）
  - FILL_BLOCK: 黒/白を色ビット（bit2）で識別に確定
  - FRAME_CONTROL: bit3で8種類に整理
- **SKIP_BLOCK**: 範囲を1〜64に確定（旧「1〜65」は誤記）
- **FOR**: FOR 0（2回）・FOR 1（3回）の生成禁止を明記

### Added

- **ヘッダー完全構造**: MD5・総フレーム数・ブロックサイズフィールドを追加
- **BLOCK_STREAM（0x30）**: フレーム先頭オペコードとして定義
- **SHIFT_BITデコード手順**: X→Y順のシフト適用・端ビットパディングを明文化
- **RLE_FRAME終端検出**: ピクセルカウントによる二重保護

### Spec

- SPEC.md rev.2〜rev.6

-----

## [0.1.0] - 2026-03-10

### Added

- 初版仕様書（SPEC.md rev.1）
- SHIFT_BIT構造確定
- RLE_BLOCK分木復元
- FILL_BLOCK確定
- 基本的なエンコーダ/デコーダ実装

-----

## バージョンポリシー

```
0.x.x  Pre-release
  エンコード/デコードの完全な整合性テストが
  完了するまでこのレンジを維持する

1.0.0  Stable
  以下の条件を満たした時点でリリース:
    - エンコード→デコード→元データ一致テスト: 全パターン通過
    - LGT8F328実機でのデコード動作確認
    - SPEC.md とコードの完全な一致確認
```