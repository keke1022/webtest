import pandas as pd
import numpy as np
import torch
import torchaudio
from transformers import BertTokenizerFast, BertForTokenClassification, Wav2Vec2FeatureExtractor
import whisper_timestamped as whisper

labels = ['FP', 'RP', 'RV', 'RS', 'PW']

def run_asr(audio_file, device, model_path):
    """Return the transcribed text from the audio file

    Args:
        audio_file (_type_): audio file to transcribe
        device (_type_): device to run the model on
        model_path (_type_): path to the model to use for transcription

    Returns:
        _type_: transcribed Dataframe
    """

    # Load audio file and resample to 16 kHz
    audio, orgnl_sr = torchaudio.load(audio_file)
    audio_rs = torchaudio.functional.resample(audio, orgnl_sr, 16000)[0, :]
    audio_rs.to(device)

    # Load in Whisper model that has been fine-tuned for verbatim speech transcription
    model = whisper.load_model(model_path, device)
    # model.to(device)
    print('loaded finetuned whisper asr') 

    # Get Whisper output
    result = whisper.transcribe(model, audio_rs, language='en', beam_size=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))

    # Convert output dictionary to a dataframe
    words = []
    for segment in result['segments']:
        words += segment['words']
    text_df = pd.DataFrame(words)
    text_df['text'] = text_df['text'].str.lower()

    # text_df is made of:
    #       text    start   end     confidence
    # 0     okay    0.00    1.98    0.488
    # 1     uh      1.98    12.38   0.209

    return text_df

def run_language_based(audio_file, text_df, device):
    """Run the language-based model on the transcribed text

    Args:
        audio_file (_type_): audio file to transcribe
        text_df (_type_): transcribed text
        device (_type_): device to run the model on

    Returns:
        (Tensor, Tensor): frame-level embeddings and predictions
    """

    # Tokenize the text
    text = ' '.join(text_df['text'])
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)

    # Initialize Bert model and load in pre-trained weights
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=5)
    model.load_state_dict(torch.load('demo_models/language.pt', map_location='cpu'), strict=False)
    print('loaded finetuned language model') 

    model.config.output_hidden_states = True
    model.to(device)

    # Get Bert output at the word-level
    output = model.forward(input_ids=input_ids)
    probs = torch.sigmoid(output.logits)
    preds = (probs > 0.5).int()[0][1:-1]
    emb = output.hidden_states[-1][0][1:-1]

    # Convert Bert word-level output to a dataframe with word timestamps
    pred_columns = [f"pred{i}" for i in range(preds.shape[1])]
    pred_df = pd.DataFrame(preds.cpu(), columns=pred_columns)
    emb_columns = [f"emb{i}" for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb.detach().cpu(), columns=emb_columns)
    df = pd.concat([text_df, pred_df, emb_df], axis=1)

    # Convert dataframe to frame-level output
    frame_emb, frame_pred = convert_word_to_framelevel(audio_file, df)

    return frame_emb, frame_pred

def convert_word_to_framelevel(audio_file, df):
    """Convert word-level predictions and embeddings to frame-level

    Args:
        audio_file (_type_): audio file to transcribe
        df (_type_): word-level predictions and embeddings

    Returns:
        (Tensor, Tensor): frame-level embeddings and predictions
    """

    # How long does the frame-level output need to be?
    df['end'] = df['end'] + 0.01
    info = torchaudio.info(audio_file)
    end = info.num_frames / info.sample_rate

    # Initialize lists for frame-level predictions and embeddings (every 10 ms)
    frame_time = np.arange(0, end, 0.01).tolist()
    num_labels = len(labels)
    frame_pred = [[0] * num_labels] * len(frame_time)
    frame_emb = [[0] * 768] * len(frame_time)

    # Loop through text to convert each word's predictions and embeddings to the frame-level (every 10 ms)
    for idx, row in df.iterrows():
        start_idx = round(row['start'] * 100)
        end_idx = round(row['end'] * 100)
        end_idx = min(end_idx, len(frame_time))
        frame_pred[start_idx:end_idx] = [[row['pred' + str(pidx)] for pidx in range(num_labels)]] * (end_idx - start_idx)
        frame_emb[start_idx:end_idx] = [[row['emb' + str(eidx)] for eidx in range(768)]] * (end_idx - start_idx)

    # Convert these frame-level predictions and embeddings from every 10 ms to every 20 ms (consistent with WavLM output)
    frame_emb = torch.Tensor(np.array(frame_emb)[::2])
    frame_pred = torch.Tensor(np.array(frame_pred)[::2])

    return frame_emb, frame_pred