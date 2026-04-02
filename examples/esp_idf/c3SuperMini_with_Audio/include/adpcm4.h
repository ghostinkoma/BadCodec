/**
 * @file  adpcm4.h
 *
 * example yt-dlp https://www.youtube.com/watch?v=FtutLA63Cp8
 * ffmpeg -i BadApple_origen.mp4 -ac 1 -ar 18000 -c:a adpcm_ima_wav 4bitadpcm.wav
 */

 #ifndef ADPCM4_H
 #define ADPCM4_H
 
 #include <stdint.h>
 

#define BAD_AUDIO_SIZE     1759326UL

 // サンプリング周波数（別配列）
 const uint32_t bad_data_sample_rate[] = {
     17000UL
 };
 
 // ADPCMデータ本体
 const uint8_t bad_audio_data[] = {
}
