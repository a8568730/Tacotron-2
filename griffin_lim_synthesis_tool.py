import numpy as np
from datasets.audio import *
import os
from hparams import hparams

out_dir = 'wav_out'

os.makedirs(out_dir, exist_ok=True)

mel_file = 'mel-batch_1_sentence_0.npy'#'training_data/mels/mel-LJ001-0005.npy'
mel_spectro = np.load('tacotron_output/eval/' + mel_file)
wav = inv_mel_spectrogram(mel_spectro.T, hparams) 
#save the wav under test_<folder>_<file>
save_wav(wav, os.path.join(out_dir, 'test_mel_{}.wav'.format(mel_file.replace('/', '_').replace('\\', '_').replace('.npy', ''))),
        sr=hparams.sample_rate)



