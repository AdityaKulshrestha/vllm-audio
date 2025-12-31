# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
import torch 
import wave
import os
import re
from scipy.io.wavfile import write
from snac import SNAC
import numpy as np


model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

snac_device = os.environ.get("SNAC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
model = model.to(snac_device)


prompts = [
  "नमस्ते, मेरा नाम है",
  "संयुक्त राज्य अमेरिका के राष्ट्रपति हैं",
  "फ़्रांस की राजधानी है",
  "एआई का भविष्य है"
]


# Reference - https://github.com/canopyai/Orpheus-TTS/blob/e64661fe6d02c414fc77c53578c9d64082614861/orpheus_tts_pypi/orpheus_tts/engine_class.py#L98C78-L98C171
# Create a sampling params object.
sampling_params = SamplingParams(
        temperature=0.9, 
        top_p=0.9, 
        max_tokens=1200, 
        stop_token_ids = [128258], 
        repetition_penalty=1.1
    )


def main():
    # Create an LLM.
    llm = LLM(model="canopylabs/3b-hi-ft-research_release", enforce_eager=True)  # For faster loading
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_tokens = output.outputs[0].token_ids

        audio_tokens = convert_token_to_audio_token(generated_tokens)
        audio = redistribute_codes(audio_tokens)
        audio_np = audio.detach().cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        write(f"output_audio_direct_{prompts.index(prompt)}.wav", 24000, audio_int16[0,0,:])

        # SOME WORKING ISSUES, CURRENTLY USING THE TOKEN IDS
        # audio_tokens =  re.findall(r'<custom_token_\d+>', generated_text)
        # audio_token_ids = [turn_token_into_id(token, idx) for idx, token in enumerate(audio_tokens)]
        # # breakpoint()
        # audio_bytes_list = [convert_to_audio(audio_token_ids[i: i+28]) for i in range(0, len(audio_token_ids), 28)]
        # save_audio_to_wav(audio_bytes_list, f"output_audio_{prompts.index(prompt)}.wav")
    
        # Check and decode if they are correct
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


################################################################################################################

def convert_token_to_audio_token(tokens_list):
    token_to_find = 128257
    token_to_remove = 128258
    audio_tokens = torch.tensor(tokens_list)

    masked_row = audio_tokens[audio_tokens != token_to_remove]

    row_length = masked_row.size(0)
    new_length = (row_length // 7) * 7
    trimmed_row = masked_row[:new_length]
    trimmed_row = [t - 128266 for t in trimmed_row]

    return trimmed_row



def turn_token_into_id(token_string, index):
    # Strip whitespace
    token_string = token_string.strip()

    # Process the last token
    if token_string.startswith("<custom_token_") and token_string.endswith(">"):
        try:
            number_str = token_string[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    else:
        return None


def redistribute_codes(code_list):
  layer_1 = []
  layer_2 = []
  layer_3 = []
  for i in range((len(code_list)+1)//7):
    layer_1.append(code_list[7*i])
    layer_2.append(code_list[7*i+1]-4096)
    layer_3.append(code_list[7*i+2]-(2*4096))
    layer_3.append(code_list[7*i+3]-(3*4096))
    layer_2.append(code_list[7*i+4]-(4*4096))
    layer_3.append(code_list[7*i+5]-(5*4096))
    layer_3.append(code_list[7*i+6]-(6*4096))
  codes = [torch.tensor(layer_1).unsqueeze(0),
         torch.tensor(layer_2).unsqueeze(0),
         torch.tensor(layer_3).unsqueeze(0)]
  audio_hat = model.decode(codes)
  return audio_hat


def convert_to_audio(multiframe):
    if len(multiframe) < 7:
        return

    codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)

    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]

    for j in range(num_frames):
        i = 7*j
        if codes_0.shape[0] == 0:
            codes_0 = torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)
        else:
            codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)])

        if codes_1.shape[0] == 0:
            
            codes_1 = torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
        else:
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)])
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])

        if codes_2.shape[0] == 0:
            codes_2 = torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])
        else:
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
    # check that all tokens are between 0 and 4096 otherwise return *
    if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
        return

    with torch.inference_mode():
        audio_hat = model.decode(codes)

    audio_slice = audio_hat[:, :, 2048:4096]
    detached_audio = audio_slice.detach().cpu()
    audio_np = detached_audio.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    return audio_bytes


def save_audio_to_wav(audio_bytes_list, filename):
    # Write a function to store the audio bytes into a wav file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)      # int16 = 2 bytes
        wf.setframerate(24000)
        wf.writeframes(b''.join(audio_bytes_list))

if __name__ == "__main__":
    main()
