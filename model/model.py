# model/model.py
import torch
import torch.nn as nn
import torchaudio
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

class AcousticModel(nn.Module):
    def __init__(self):
        super(AcousticModel, self).__init__()
        self.basemodel = WavLMModel.from_pretrained('microsoft/wavlm-base')
        self.linear = nn.Linear(768, 5)

    def forward(self, x):
        feats = self.basemodel.feature_extractor(x)
        feats = feats.transpose(1, 2)
        feats, _ = self.basemodel.feature_projection(feats)
        emb = self.basemodel.encoder(feats, return_dict=True)[0]
        out = self.linear(emb)

        return emb, out

class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()

        self.hidden_size = 512
        self.blstm = nn.LSTM(input_size=768 * 2,
                             hidden_size=self.hidden_size,
                             num_layers=1,
                             batch_first=True,
                             bidirectional=True)
        self.fc = nn.Linear(self.hidden_size*2, 5)

    def forward(self, x_bert, x_w2v2):

        x_cat = torch.cat((x_bert, x_w2v2), dim=-1)
        x_cat, _ = self.blstm(x_cat)

        out = self.fc(x_cat)

        return out

def load_model(path):
    if ('acoustic' in path):
        model = AcousticModel()
    elif ('multimodal' in path):
        model = MultimodalModel()
    else:
        raise ValueError('Model type not supported')

    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.to("cpu")
    print('loaded finetuned model')
    return model

def predict_acoustic(model, audio_file):
    # Load audio file and resample to 16 kHz
    audio, orgnl_sr = torchaudio.load(audio_file)
    audio_rs = torchaudio.functional.resample(audio, orgnl_sr, 16000)[0, :]
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False
    )
    audio_feats = feature_extractor(audio_rs, sampling_rate=16000).input_values[0]
    audio_feats = torch.Tensor(audio_feats).unsqueeze(0)
    audio_feats = audio_feats.to('cpu')
    
    # Get WavLM output
    emb, output = model(audio_feats)
    probs = torch.sigmoid(output)
    preds = (probs > 0.5).int()[0]
    emb = emb[0]

    return emb, preds

def predict_multimodal(model, acoustic_emb, language_emb, device):
    min_size = min(language_emb.size(0), acoustic_emb.size(0))
    acoustic = acoustic_emb[:min_size].unsqueeze(0)
    language = language_emb[:min_size].unsqueeze(0)

    language = language.to(device)
    acoustic = acoustic.to(device)

    # Initialize multimodal model and load in pre-trained weights
    model = MultimodalModel()
    model.load_state_dict(torch.load('demo_models/multimodal.pt', map_location='cpu'))
    model.to(device)
    print('loaded finetuned multimodal model') 

    # Get multimodal output
    output = model(language, acoustic)
    probs = torch.sigmoid(output)
    preds = (probs > 0.5).int()[0]

    return preds
