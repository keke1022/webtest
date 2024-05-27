from flask import Flask, request, jsonify, render_template
from asr_module import run_asr, run_language_based
from basic_module import align_word_preds
from model.model import load_model, predict_acoustic, predict_multimodal

app = Flask(__name__)
model_fast = load_model('demo_models/acoustic.pt')
model_slow = load_model('demo_models/multimodal.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_disfluency():
    device = 'mps'
    file = request.files['file']
    mode = request.form['mode']

    # Save the uploaded file to a temporary location
    audio_path = "temp_audio.wav"
    file.save(audio_path)

    text_df = None
    text_df = run_asr(audio_path, device, 'demo_models/asr')
    language_emb, _ = run_language_based(audio_path, text_df, device)

    # Choose the model and prediction function based on the selected mode
    acoustic_emb, preds = predict_acoustic(model_fast, audio_path)

    if mode == 'slow':
        preds = predict_multimodal(model_slow, acoustic_emb, language_emb, device)

    # Convert predictions and embeddings to lists for JSON serialization
    preds_list = preds.tolist()
    # emb_list = emb.tolist()

    text_df = align_word_preds(text_df, preds_list)

    result = {
        'text': text_df['text'].tolist(),
        'FP': text_df['FP'].tolist(), 
        'RP': text_df['RP'].tolist(), 
        'RV': text_df['RV'].tolist(), 
        'RS': text_df['RS'].tolist(), 
        'PW': text_df['PW'].tolist()
        # 'pred': text_df['pred'].tolist()
        # "predictions": preds_list,
    }
    print(result)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)