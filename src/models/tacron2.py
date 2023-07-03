import torch
import torch.nn as nn
from torch.nn import functional
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.prenet_postnet import Postnet


class Tacotron2(nn.Module):
    def __init__(self, config):
        super(Tacotron2, self).__init__()


        self.mel_spec_channels = config.number_mel_channels
        self.num_f_p_s = config.frames_per_step_num

        self.emb = nn.Embedding(
            config.sympols_number, config.emb_dim_symbols)

        standard_deviation = pow(
            2 / float(config.emb_dim_symbols + config.sympols_number), .5)

        mean_by_std = standard_deviation * pow(3.0, .5)

        self.emb.weight.data.uniform_(-mean_by_std, mean_by_std)

        self.postnet = Postnet(config)
        self.decoder = Decoder(config)
        self.encoder = Encoder(config)

    def parser_out(self, results, output_len=None):

        maximum_length = results[0].size(-1)
        temp = torch.arange(0, maximum_length).to(output_len.device)
        output_mask = ~((temp < output_len.unsqueeze(1)).bool())

        output_mask = output_mask.expand(
            self.mel_spec_channels, output_mask.size(0), output_mask.size(1))
        output_mask = output_mask.transpose(0, 1)

        results[2].data.masked_fill_(output_mask[:, 0, :], 1e3)
        results[1].data.masked_fill_(output_mask, 0.0)
        results[0].data.masked_fill_(output_mask, 0.0)

        return results

    def forward(self, texts, text_len, melspectograms, result_len):

        input_emb = self.emb(texts).permute(0, 2, 1)

        enc_out = self.encoder(input_emb, text_len)

        melspec_out, decoder_outputs, alignments = self.decoder(enc_out, melspectograms, input_lengths=text_len)

        decoder_outputs = (decoder_outputs.unsqueeze(
            2).repeat(1, 1,  self.num_f_p_s)).view(decoder_outputs.size(0), -1)

        postnet_out = self.postnet(melspec_out)
        postnet_out += melspec_out

        return self.parser_out([melspec_out, postnet_out, decoder_outputs, alignments], result_len)

    def inference(self, texts):

        input_emb = self.emb(texts).permute(0, 2, 1)
        enc_out = self.encoder.inference(input_emb)
        melspec_out, decoder_outputs, alignments = self.decoder.inference(
            enc_out)

        postnet_out = self.postnet(melspec_out)
        postnet_out += melspec_out

        return [melspec_out, postnet_out,
                decoder_outputs, alignments]



