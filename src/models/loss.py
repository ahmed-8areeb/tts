from torch import nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_func_result = nn.BCEWithLogitsLoss()
        self.loss_func_postnet = nn.MSELoss()
        self.loss_func_dec = nn.MSELoss()

    def forward(self, final_result, true_out):
        mel_true_result, postnet_true_result = true_out[0], true_out[1]
        postnet_true_result = postnet_true_result.view(-1, 1)

        mel_predicted_result, postnet_predicted_result, predictd_result, _ = final_result
        predictd_result = predictd_result.view(-1, 1)

        
        final_out_loss = self.loss_func_result(predictd_result, postnet_true_result)
        posnet_loss = self.loss_func_postnet(postnet_predicted_result, mel_true_result)
        melspectogram_loss = self.loss_func_dec(mel_predicted_result, mel_true_result)
        return melspectogram_loss + posnet_loss + final_out_loss
