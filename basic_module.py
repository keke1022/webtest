import numpy as np
import pdbp

def align_word_preds(text_df, preds):
    for idx, row in text_df.iterrows():
        start_idx = round(row['start'] * 50)
        end_idx = round(row['end'] * 50)
        word_preds = np.array(preds[start_idx:end_idx])
        print(
            'start index: {}'.format(start_idx),
            'end index: {}'.format(end_idx),
            'text: {}'.format(row['text']),
            'word preds: {}'.format(word_preds)
        )
        # breakpoint()
        mean_preds = np.mean(word_preds, axis=0)
        for c, pred in enumerate(['FP', 'RP', 'RV', 'RS', 'PW']):
            text_df.loc[idx, pred] = mean_preds[c]
    return text_df
