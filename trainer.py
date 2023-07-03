import time
import shutil
import os
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from data.dataset import Dataset, OrganizeData
from src.models.loss import Loss
from src.optimizer.scheduler import WarmupLR
from src.predictor import convert_dict_to_object
from src.models.tacron2 import Tacotron2

configs_path = "configs/config.yml"

assert (torch.cuda.is_available()), 'No GPU available :('
device = torch.device('cuda:0')

if os.path.exists(configs_path):
    f = open(configs_path, 'r', encoding='utf-8')
    configs = convert_dict_to_object(yaml.load(f, Loader=yaml.FullLoader))
    f.close()
else:
    raise ValueError('No config file :(')

######################## Configs ########################
frames_per_step_num = configs.config_model.frames_per_step_num
train_text = configs.config_dataset.training_data_path
mel_dir = configs.config_dataset.mel_folder_path
batch_size = configs.config_training.batch_size
workers_number = configs.config_training.workers_number
learning_rate = configs.config_optimizer.learning_rate
weight_decay = configs.config_optimizer.weight_decay
config_schedular = configs.config_optimizer.config_schedular
maximum_number_epochs = configs.config_training.maximum_number_epochs
clip_grad = configs.config_training.clip_grad
pretrained_model = "models/Tacotron2/best_model"
save_model_path = "models/"
#########################################################

collate_fn = OrganizeData(frames_per_step_num)
train_dataset = Dataset(train_text, mel_dir)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           collate_fn=collate_fn,
                                           shuffle=True,
                                           num_workers=workers_number,
                                           drop_last=False)

model = Tacotron2(configs.config_model)
model.to(device)
criterion = Loss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=float(
    learning_rate), weight_decay=float(weight_decay))
scheduler = WarmupLR(
    optimizer=optimizer, **config_schedular)


# if pretrained_model exists, load it:
if os.path.exists(pretrained_model):
    print("Loading pretrained model...")
    pretrained_model = os.path.join(pretrained_model, 'model.pt')

    model_dict = model.state_dict()
    model_state_dict = torch.load(pretrained_model)

    for name, weight in model_dict.items():
        if name in model_state_dict.keys():
            if list(weight.shape) != list(model_state_dict[name].shape):
                model_state_dict.pop(name, None)

    model.load_state_dict(model_state_dict, strict=False)


def save_model(epoch_id, test_loss, best_model=False):
    save_model_name = 'Tacotron2'
    if best_model:
        model_path = os.path.join(
            save_model_path, save_model_name, 'best_model')
    else:
        model_path = os.path.join(
            save_model_path, save_model_name, f'epoch_{epoch_id}')

    os.makedirs(model_path, exist_ok=True)

    # save optimizer
    torch.save(optimizer.state_dict(),
               os.path.join(model_path, 'optimizer.pt'))

    # save model
    torch.save(model.state_dict(),
               os.path.join(model_path, 'model.pt'))

    # save model state
    with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
        f.write(f'{{"last_epoch": {epoch_id}, "test_loss": {test_loss}}}')

    # only save the models of the last 3 epochs
    if not best_model:
        last_model_path = os.path.join(
            save_model_path, save_model_name, 'last_model')
        shutil.rmtree(last_model_path, ignore_errors=True)
        shutil.copytree(model_path, last_model_path)

        old_model_path = os.path.join(
            save_model_path, save_model_name, 'epoch_{epoch_id - 3}')
        if os.path.exists(old_model_path):
            shutil.rmtree(old_model_path)

###################################################### TRAIN ######################################################


def train_epoch(epoch_id):
    batch_losses = []
    model.train()

    for batch_id, batch in enumerate(tqdm(train_loader, desc=f'epoch:{epoch_id}')):
        text_padded, text_lengths, target_mel, target_gate, mel_lengths = batch

        text_padded = text_padded.to(device)
        text_lengths = text_lengths.to(device)
        target_mel = target_mel.to(device)
        target_gate = target_gate.to(device)
        mel_lengths = mel_lengths.to(device)

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(text_padded, text_lengths, target_mel, mel_lengths)
            loss = criterion(outputs, [target_mel, target_gate])

        loss = loss
        batch_losses.append(loss.cpu().detach().numpy())
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_grad)
        if torch.isfinite(grad_norm):
            optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if batch_id % 100 == 0:
            print(f'epoch:{epoch_id}, batch:{batch_id}, loss:{loss}')
    return float(sum(batch_losses) / len(batch_losses))

best_error_rate = 1e3
for epoch_id in range(maximum_number_epochs):
    epoch_id += 1
    start_epoch = time.time()
    epoch_loss = train_epoch(epoch_id=epoch_id)

    if epoch_loss < best_error_rate:
        best_error_rate = epoch_loss
        save_model(epoch_id=epoch_id, test_loss=epoch_loss, best_model=True)

    save_model(epoch_id=epoch_id, test_loss=epoch_loss)
