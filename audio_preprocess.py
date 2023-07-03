# do this

import argparse
import glob
import os

import numpy as np
import librosa
from tqdm import tqdm

from src.utils.utils import print_arguments
from src.data_utils.utils import pinyin_2_phoneme


def wav2feature(wav_file, args):
    wav, _ = librosa.load(wav_file, sr=None, mono=True)
    fbank = librosa.feature.melspectrogram(y=wav,
                                           sr=args.fs,
                                           fft_number=args.fft_number,
                                           window_len=args.window_len,
                                           lenght_hop=args.lenght_hop,
                                           n_mels=args.n_mels,
                                           min_f=args.min_f,
                                           max_f=args.max_f)
    log_fbank = librosa.power_to_db(fbank, ref=np.max)
    return log_fbank


def processing_wavs(wav_files, args):
    feats = []
    ids = []
    for file in tqdm(wav_files, desc='featurizer'):
        id_wav = os.path.split(file)[-1][:-4]
        fea = wav2feature(file, args)
        feats.append(fea)
        ids.append(id_wav)

    # 计算特征的均值和方差
    fea_array = np.concatenate(feats, axis=1)  # fea的维度 D*T
    fea_mean = np.mean(fea_array, axis=1, keepdims=True)
    fea_std = np.std(fea_array, axis=1, keepdims=True)

    mel_save_path = os.path.join(args.output_dir, 'mel_features')
    os.makedirs(mel_save_path, exist_ok=True)

    # 对所有的特征进行正则, 并保存
    for feat, id_wav in zip(feats, ids):
        norm_fea = (feat - fea_mean) / fea_std
        fea_name = os.path.join(mel_save_path, id_wav + '.npy')
        np.save(fea_name, norm_fea)

    static_name = os.path.join(mel_save_path, 'static.npy')
    np.save(static_name, np.array([fea_mean, fea_std], dtype=object))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    fs = 22050
    parser.add_argument('--input_dir', type=str, default='./data', help='raw data dir')
    parser.add_argument('--output_dir', type=str, default='./data', help='data output dir')
    parser.add_argument('--dictionary_file_path', type=str, default='./data/vocab', help='vocabulary path')
    parser.add_argument('--fs', type=int, default=fs, help='sampling_rate')
    parser.add_argument('--fft_number', type=int, default=4096)
    parser.add_argument('--window_len', type=int, default=int(fs * 0.05))
    parser.add_argument('--lenght_hop', type=int, default=int(fs * 0.0125))
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--min_f', type=float, default=0.0)
    parser.add_argument('--max_f', type=float, default=fs / 2)
    args = parser.parse_args()
    print_arguments(args=args)

    waves = glob.glob(r"D:\Ziyad\Voice DataSets\LJSpeech-1.1\wavs\*.wav")
    processing_wavs(waves, args)
    # trans_prosody(args)
