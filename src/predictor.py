import os
import text_preprocessing
import librosa
import numpy as np
import torch
import yaml
import soundfile as sf
from src.models.tacron2 import Tacotron2
from src.enhancment.enhancement import spectral_subtraction

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def convert_dict_to_object(dict_obj):
    if not isinstance(dict_obj, dict):
        return dict_obj
    inst = Dict()
    for k, v in dict_obj.items():
        inst[k] = convert_dict_to_object(v)
    return inst

class Tacotron2Predictor:
    def __init__(self,
                config_file="configs/config.yml",
                path_for_model='models/Tacotron2',
                gpu=True):

        with open(config_file, encoding='utf-8') as file:
            config_file = yaml.load(file.read(), Loader=yaml.FullLoader)

        
        self.config = convert_dict_to_object(config_file)
        
        if gpu is True:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
  
        
        path_for_model += '/female_voice_best_model.pt'
        self.model = torch.load(path_for_model, map_location='cpu')
        # self.model = Tacotron2(self.config.config_model)
        # dict_model_state = torch.load(path_for_model, map_location='cpu')
        # self.model.load_state_dict(dict_model_state)
        self.model.to(self.device)
        self.model.eval()


    def predict(self, line, path, noise_removal=False):

        
        id_dictionary = text_preprocessing.get_symbol_index()
        
        words_dictionary = text_preprocessing.get_set_words()

        ids = []
        
        line = text_preprocessing.preprocess_sent(line)
        
        
        for word in line:
            if word not in words_dictionary:
                continue
            for phoneme in words_dictionary[word]:
                ids.append(id_dictionary[phoneme])


        torch_text = torch.tensor(ids)
        torch_text = torch_text.unsqueeze(0).to(self.device)

        #  at evaluation 
        with torch.no_grad():
            evaluation_out = self.model.inference(torch_text)
            melspec_out = evaluation_out[1].squeeze(0).cpu().detach().numpy()

        f = self.configs.config_dataset.mel_folder_path + 'static.npy'
        
        melspec_static = np.load(f)

        mel_mean = melspec_static[0]
        mel_std = melspec_static[1]
        mel_mean = np.float64(mel_mean)
        mel_std = np.float64(mel_std)
        
        predicted_mel =  mel_mean + melspec_out * mel_std 

# 

        inv_fbank = librosa.db_to_power(predicted_mel)
        inv_wav = librosa.feature.inverse.mel_to_audio(inv_fbank,
                                                      sr=self.configs.config_preprocess.sample_rate,
                                                      fft_number=self.configs.config_preprocess.fft_number,
                                                      window_len=self.configs.config_preprocess.window_len,
                                                      lenght_hop=self.configs.config_preprocess.lenght_hop,
                                                      min_f=self.configs.config_preprocess.min_f,
                                                      max_f=self.configs.config_preprocess.max_f)
        inv_wav = inv_wav / max(inv_wav)
        # if noise_removal:
        #     inv_wav = speech_enhance(wave_data=inv_wav,
        #                             fft_number=self.configs.config_preprocess.fft_number,
        #                             lenght_hop=self.configs.config_preprocess.lenght_hop,
        #                             window_len=self.configs.config_preprocess.window_len,
        #                             noise_frame=30)
        inv_wav, _ = librosa.effects.trim(inv_wav)
        sf.write(path, inv_wav, self.configs.config_preprocess.sample_rate)
