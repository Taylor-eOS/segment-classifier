import os
import main

model_path = os.path.join('model', 'combined_model.h5')
main.infer('inference_input.wav', model_path)
