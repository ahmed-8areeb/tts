import torch
import torch.nn as nn
from torch.nn import functional
from src.models.utilities import LinearTransformation
from src.models.prenet_postnet import Prenet
from src.models.attention import Attention
# import Variable
from torch.autograd import Variable as Var


def lengths_mask(len):
    maximum_length = torch.max(len).item()
    temp = torch.arange(0, maximum_length).to(len.device)
    return (temp < len.unsqueeze(1)).bool()


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.mel_spec_channels = config.number_mel_channels
        self.embedding_encoder_dim = config.emb_dim_encoder
        self.num_f_p_s = config.frames_per_step_num
        self.rnn_decoder_dim = config.rnn_dim_decoder
        self.rnn_att_dim = config.rnn_atten_dim
        self.dec_max_stps = config.decoder_max_steps
        self.pre_dim = config.dim_predent
        self.pDecoderDropout = config.decoder_dropout_probabily
        self.pAttenDropout = config.attention_dropout_probabily
        self.convergec_thresh = config.convergenc_threshold

        self.rnn_attention = nn.LSTMCell(
            config.dim_predent + config.emb_dim_encoder,
            config.rnn_atten_dim)

        self.prenet = Prenet(
            config.frames_per_step_num * config.number_mel_channels,
            [config.dim_predent, config.dim_predent])

        self.atten_lay = Attention(
            out_att_dim=config.atten_dim, in_channels=config.numbre_filters_attention_location,
            rnn_hidden_dim=config.rnn_atten_dim, in_emb_dim=config.emb_dim_encoder,
            kernel_size=config.kernal_size_attention_location)

        self.projection = LinearTransformation(
            input_dim=(config.emb_dim_encoder + config.rnn_dim_decoder),
            output_dim=(config.frames_per_step_num * config.number_mel_channels))

        self.rnn_decoder = nn.LSTMCell(
            config.rnn_atten_dim + config.emb_dim_encoder,
            config.rnn_dim_decoder, 1)
        self.rnn_decoder_drop = nn.Dropout(0.1)

        self.convergence_layer = LinearTransformation(
            input_dim=(config.rnn_dim_decoder + config.emb_dim_encoder), output_dim=1,
            bias=True, weight_init='sigmoid')

    def masked_frame(self, input):
        masked = input.size(0)
        return Var(input.data.new(
            masked, self.mel_spec_channels * self.num_f_p_s).zero_())

    def initialization(self, input, masked_input):

        self.input = input
        time = input.size(1)
        temp = input.size(0)

        self.cell_attention = Var(input.data.new(
            temp, self.rnn_att_dim).zero_())

        self.cell_decoder = Var(input.data.new(
            temp, self.rnn_decoder_dim).zero_())

        self.hidden_attention = Var(input.data.new(
            temp, self.rnn_att_dim).zero_())

        self.hidden_decoder = Var(input.data.new(
            temp, self.rnn_decoder_dim).zero_())

        self.weights_attention_cumulative = Var(input.data.new(
            temp, time).zero_())

        self.weights_attention = Var(input.data.new(
            temp, time).zero_())

        self.context_attention = Var(input.data.new(
            temp, self.embedding_encoder_dim).zero_())

        self.masked_input = masked_input

        self.processed_input = self.atten_lay.encoder_out_layer(input)

    
    def decoder_inputs_parser(self, input):
        input = input.permute(0, 2, 1)
        time = input.size(1)
        temp = input.size(0)
        input = input.reshape(temp, time // self.num_f_p_s, -1)

        return input.permute(1, 0, 2)

    
    def decoder_outputs_parser(self, melspectogram, output, output_alignments):

        output_alignments = torch.stack(output_alignments).permute(1, 0, 2)
        output = torch.stack(output).permute(1, 0).contiguous()

        melspectogram = torch.stack(melspectogram).permute(1, 0, 2)
        melspectogram = melspectogram.contiguous()
        melspectogram = melspectogram.reshape(
            melspectogram.size(0), -1,self.mel_spec_channels)
        melspectogram = melspectogram.permute(0, 2, 1)
        return melspectogram, output, output_alignments

    def decoder(self, input_for_decoder):

        torch_input = torch.cat(
            (input_for_decoder, self.context_attention), -1)

        self.hidden_attention, self.cell_attention = self.rnn_attention(
            torch_input, (self.hidden_attention, self.cell_attention))

        self.hidden_attention = functional.dropout(
            self.hidden_attention, self.pAttenDropout, self.training)

        weights = torch.cat(
            (self.weights_attention.unsqueeze(1),
             self.weights_attention_cumulative.unsqueeze(1)), dim=1)

        self.context_attention, self.weights_attention = self.atten_lay(
            self.hidden_attention, self.input, self.processed_input,
            weights, self.masked_input)

        input_for_decoder = torch.cat(
            (self.hidden_attention, self.context_attention), -1)

        self.weights_attention_cumulative = self.weights_attention + \
            self.weights_attention_cumulative

        self.hidden_decoder, self.cell_decoder = self.rnn_decoder(
            input_for_decoder, (self.hidden_decoder, self.cell_decoder))

        self.hidden_decoder = functional.dropout(
            self.hidden_decoder, self.pDecoderDropout, self.training)

        hidden_decoder_attention = torch.cat(
            (self.hidden_decoder, self.context_attention), dim=1)

        output_for_decoder = self.projection(
            hidden_decoder_attention)

        output_prob = self.convergence_layer(hidden_decoder_attention)

        return output_for_decoder, output_prob, self.weights_attention

    def forward(self, input, inputs_to_decoder, input_lengths):

        melspectogram = []
        output = []
        output_alignments = []

        inputs_to_decoder = self.decoder_inputs_parser(inputs_to_decoder)
        decoder_input = self.masked_frame(input).unsqueeze(0)

        inputs_to_decoder = torch.cat(
            (decoder_input, inputs_to_decoder), dim=0)
        inputs_to_decoder = self.prenet(inputs_to_decoder)

        # print("here ------------------------ ",input.size())
        self.initialization(
            input, masked_input=~lengths_mask(input_lengths))

        while len(melspectogram) < inputs_to_decoder.size(0) - 1:
            decoder_input = inputs_to_decoder[len(melspectogram)]
            output_mel, result, weights = self.decoder(
                decoder_input)
            melspectogram += [output_mel.squeeze(1)]
            output_alignments += [weights.squeeze(1)]
            output += [result.squeeze(1)]

        melspectogram, output, output_alignments = self.decoder_outputs_parser(
            melspectogram, output, output_alignments)

        return melspectogram, output, output_alignments

    def inference(self, input_mem):
        melspectogram = []
        output = []
        output_alignments = []

        input = self.masked_frame(input_mem)

        self.initialization(input, masked_input=None)

        while True:
            input = self.prenet(input)
            output_mel, result, alignment = self.decoder(input)

            melspectogram += output_mel.squeeze(1).tolist()
            output_alignments += alignment.squeeze(1).tolist()
            output += result.squeeze(1).tolist()

            if torch.sigmoid(result) > self.convergec_thresh:
                break

            elif len(melspectogram) == self.dec_max_stps:
                print("take care max decoder steps is reached")
                break

            input = output_mel

        melspectogram, output, output_alignments = self.decoder_outputs_parser(
            melspectogram, output, output_alignments)

        return melspectogram, output, output_alignments
