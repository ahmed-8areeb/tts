import os
import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, path, mels_directory):
        data_files = np.loadtxt(path, dtype='str', delimiter='|')
        self.data_ids = data_files[:, 0]
        self.phonetics = data_files[:, 1]
        self.mels_directory = mels_directory

    def mels(self, id):
        file_path = os.path.join(self.mels_directory, id + '.npy')
        mels = torch.from_numpy(np.load(file_path))
        return mels


    def texts(self, phonetics):
        phonetics_id = [int(id) for id in phonetics.split()]
        return torch.IntTensor(phonetics_id)

    def mels_texts(self, path, phonetics):

        mels = self.mels(path)
        texts = self.texts(phonetics)
        return texts, mels

    def __getitem__(self, i):
        return self.mels_texts(self.data_ids[i], self.phonetics[i])

    def __len__(self):
        return len(self.data_ids)


class OrganizeData:

    def __init__(self, n_f_p_s):
        self.number_framesPerStep = n_f_p_s

    def __call__(self, batch):
        in_len, sorted_ids = torch.sort(
            torch.LongTensor([len(temp[0]) for temp in batch]),
            dim=0, descending=True)
        in_len_max = in_len[0]
        
        mels_len = batch[0][1].size(0)
        length_max = 0
        for i in range(len(sorted_ids)):
            length_max = max(length_max, batch[sorted_ids[i]][1].size(1))

        mod = length_max % self.number_framesPerStep

        if mod != 0:
            length_max = length_max + self.number_framesPerStep - mod

        padded_result = torch.FloatTensor(
            len(batch), length_max).zero_()

        padded_mel = torch.FloatTensor(
            len(batch), mels_len, length_max).zero_()

        out_len = torch.LongTensor(len(batch))

        for i, id in enumerate(sorted_ids):
            melspectogram = batch[id][1]
            padded_mel[i, :, :melspectogram.size(1)] = melspectogram
            out_len[i] = melspectogram.size(1)
            padded_result[i, melspectogram.size(1) - 1:] = 1

        padded_text = torch.LongTensor(len(batch), in_len_max)
        padded_text.zero_()
        for i,id in enumerate(sorted_ids):
            data = batch[id][0]
            padded_text[i, :data.size(0)] = data

        return padded_text, in_len, padded_mel, padded_result, out_len
