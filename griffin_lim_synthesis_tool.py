import numpy as np
from datasets.audio import *
from sys import argv
from hparams import hparams

# After Tacotron-2 training, 
# if you only want to run Tacotron synthesis, check the script below:
#
# 1. Run Tacotron model evaluation
# $ python synthesize.py --model='Tacotron' --mode='eval' --name='Tacotron-2' --text_list ./custom_test_sentence.txt
#
# 2. Output audio
# $ python griffin_lim_synthesis_tool.py tacotron_output/eval/mel-batch_0_sentence_0.npy wav_out/test_0.wav
#

mel_file = argv[1]
output_file = argv[2]

mel_spectro = np.load(mel_file)
wav = inv_mel_spectrogram(mel_spectro.T, hparams) 
#save the wav under test_<folder>_<file>
save_wav(wav, output_file, sr=hparams.sample_rate)



