# Copyright 2024 Rich Cannings <rcannings@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import whisper
# import google.generativeai as genai # No longer directly used in main
import colorama
import threading
import sounddevice as sd
from scipy.signal import resample
import requests # For sending data to Flask app
import json # For formatting JSON payloads

from constants import *
from gemini_manager import GeminiManager # Import the new class
from app import run_flask_app # Import the function to run Flask

# Whisper Initialization
whisper_model = whisper.load_model(WHISPER_MODEL)

# Initialize GeminiManager (this will handle API key loading and model setup)
gemini_manager = GeminiManager()

# URL for the Flask app (running locally)
FLASK_APP_URL = "http://localhost:5000"

# Audio Processing Thread
def process_audio_thread(raw_audio_data, capture_samplerate):
    """Processes recorded audio in a separate thread."""
    if raw_audio_data is None or raw_audio_data.size == 0:
        print(colorama.Fore.CYAN + "No audio data received for processing.")
        return

    try:
        if raw_audio_data.dtype != np.float32:
            print(colorama.Fore.YELLOW + f"Warning: Audio data is {raw_audio_data.dtype}, converting to float32.")
            processed_audio = raw_audio_data.astype(np.float32)
        else:
            processed_audio = raw_audio_data

        if capture_samplerate != WHISPER_TARGET_SAMPLE_RATE:
            num_samples = int(len(processed_audio) * WHISPER_TARGET_SAMPLE_RATE / capture_samplerate)
            processed_audio = resample(processed_audio, num_samples)
            current_samplerate = WHISPER_TARGET_SAMPLE_RATE
            print(colorama.Fore.MAGENTA + f"Resampled audio from {capture_samplerate} Hz to {current_samplerate} Hz.")
        else:
            current_samplerate = capture_samplerate

        min_duration_for_transcription = 1.0
        if len(processed_audio) < WHISPER_TARGET_SAMPLE_RATE * min_duration_for_transcription:
            print(colorama.Fore.CYAN + f"Audio too short ({len(processed_audio)/WHISPER_TARGET_SAMPLE_RATE:.2f}s) for transcription.")
            return

        print(colorama.Fore.MAGENTA +
              f"Audio for Whisper - Dtype: {processed_audio.dtype}, Shape: {processed_audio.shape}, Rate: {current_samplerate}, " +
              f"Min: {np.min(processed_audio):.4f}, Max: {np.max(processed_audio):.4f}, Mean: {np.mean(processed_audio):.4f}")

        transcript = whisper_model.transcribe(processed_audio, fp16=True, language='en')
        transcribed_text = transcript['text'].strip()
        print(colorama.Fore.YELLOW + "Transcription: " + transcribed_text)

        # Send transcription to Flask app
        try:
            requests.post(f"{FLASK_APP_URL}/add_transcription", json={"text": transcribed_text}, timeout=2)
        except requests.exceptions.RequestException as e:
            print(colorama.Fore.RED + f"Failed to send transcription to Flask app: {e}")

        if transcribed_text:
            processed_gemini_output_str = gemini_manager.get_structured_output(transcribed_text)
            if processed_gemini_output_str:
                print(colorama.Fore.GREEN + "Gemini Output (raw string):")
                print(colorama.Fore.GREEN + processed_gemini_output_str)
                
                should_send_to_ui = True
                try:
                    # Parse the JSON string from Gemini
                    gemini_data = json.loads(processed_gemini_output_str)
                    
                    # Check if callsign, name, AND location are all "unknown" (case-insensitive)
                    callsign = gemini_data.get("callsign", "").lower()
                    name = gemini_data.get("name", "").lower()
                    location = gemini_data.get("location", "").lower()
                    
                    if callsign == "unknown" and name == "unknown" and location == "unknown":
                        should_send_to_ui = False
                        print(colorama.Fore.CYAN + "Skipping update to Detected Call Signs: callsign, name, and location are all unknown.")
                        
                except json.JSONDecodeError as e:
                    print(colorama.Fore.RED + f"Error decoding Gemini JSON output: {e}")
                    # If JSON is malformed, it won't match the filter. 
                    # The original code would have sent it; we'll maintain that behavior.
                    # Alternatively, one could set should_send_to_ui = False here.

                if should_send_to_ui:
                    # Send original Gemini output string to Flask app
                    try:
                        requests.post(f"{FLASK_APP_URL}/add_gemini_output", json={"text": processed_gemini_output_str}, timeout=2)
                    except requests.exceptions.RequestException as e:
                        print(colorama.Fore.RED + f"Failed to send Gemini output to Flask app: {e}")
            else:
                print(colorama.Fore.YELLOW + "Gemini processing returned no output or an error occurred.")
        else:
            print(colorama.Fore.CYAN + "No text from transcription to send to Gemini.")

    except Exception as e:
        print(colorama.Fore.RED + f"Error in processing thread: {e}")
        import traceback
        print(traceback.format_exc())

# Main Listening Loop
def find_audio_device_index_sd(device_name_query):
    """Finds the audio device index for sounddevice based on a partial name match."""
    devices = sd.query_devices()
    #print("Available audio devices:") # Simplified print
    matching_device_index = None
    for i, dev_info in enumerate(devices):
        # Quieter listing, only print essential info
        #print(f"  {i}: {dev_info['name']} (In: {dev_info['max_input_channels']}, Rate: {dev_info['default_samplerate']})") 
        if device_name_query.lower() in dev_info['name'].lower() and dev_info['max_input_channels'] > 0:
            # No need to print "Found matching device" here, it's implicit if returned
            matching_device_index = i
    
    if matching_device_index is None:
        try:
            # Try to use the device by name directly, will raise an exception if not valid
            sd.check_input_settings(device=device_name_query, samplerate=CAPTURE_SAMPLE_RATE, channels=CAPTURE_CHANNELS, dtype=CAPTURE_DTYPE)
            # If no exception, the device name itself is usable
            return device_name_query 
        except Exception as e:
            # Keep this warning as it's important if lookup fails
            print(colorama.Fore.YELLOW + f"Device '{device_name_query}' not directly usable by name or found by query: {e}") 
            pass # Continue to return None if not found

    if matching_device_index is not None:
        return matching_device_index
    
    return None

def main():
    colorama.init(autoreset=True)

    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    print(colorama.Fore.CYAN + "Flask web UI started in a background thread. Access at http://localhost:5000")

    target_device = find_audio_device_index_sd(DEVICE_NAME)

    if target_device is None:
        print(colorama.Fore.RED + f"Error: Audio device matching '{DEVICE_NAME}' not found or not suitable for input.")
        print("Please check your device name and ensure it is connected and has input channels.")
        return
    
    print(colorama.Fore.CYAN + f"Using device: {target_device}")
    frame_size = int(FRAME_DURATION_SECONDS * CAPTURE_SAMPLE_RATE)
    recorded_frames = []
    is_recording = False
    silence_frames_count = 0 # Renamed from silence_chunks_count for clarity with frame-based VAD
    frames_for_silence = int(NEW_SILENCE_DURATION_SECONDS / FRAME_DURATION_SECONDS)

    print(colorama.Fore.CYAN + "Listening for speech... Press Ctrl+C to stop.")

    try:
        with sd.InputStream(samplerate=CAPTURE_SAMPLE_RATE, 
                              device=target_device, 
                              channels=CAPTURE_CHANNELS, 
                              dtype=CAPTURE_DTYPE) as stream:
            while True:
                frame_data, overflowed = stream.read(frame_size)
                if overflowed:
                    print(colorama.Fore.YELLOW + "Warning: Input overflowed!")
                
                current_frame = np.squeeze(frame_data)
                if current_frame.ndim > 1 and current_frame.shape[1] > 0: # If stereo, take first channel
                    current_frame = current_frame[:,0]
                elif current_frame.ndim == 0: # if current_frame became scalar after squeeze
                    current_frame = np.array([current_frame]) # make it 1D array
                
                if current_frame.size == 0:
                    time.sleep(0.01)
                    continue

                amplitude = np.max(np.abs(current_frame))

                if amplitude > SD_SOUND_AMPLITUDE_THRESHOLD:
                    if not is_recording:
                        print(colorama.Fore.BLUE + "Sound detected, recording...")
                        is_recording = True
                        recorded_frames.clear()
                        silence_frames_count = 0
                    recorded_frames.append(current_frame)
                    silence_frames_count = 0
                elif is_recording:
                    recorded_frames.append(current_frame)
                    silence_frames_count += 1
                    if silence_frames_count >= frames_for_silence:
                        print(colorama.Fore.MAGENTA + "Silence detected, processing...")
                        if recorded_frames:
                            full_recording = np.concatenate(recorded_frames)
                            thread = threading.Thread(target=process_audio_thread, args=(full_recording.copy(), CAPTURE_SAMPLE_RATE))
                            thread.daemon = True
                            thread.start()
                        else:
                            print(colorama.Fore.CYAN + "No frames recorded for processing.")
                        is_recording = False
                        recorded_frames.clear()
                        silence_frames_count = 0
    except KeyboardInterrupt:
        print(colorama.Fore.CYAN + "\nCtrl+C received. Shutting down...")
    except Exception as e:
        print(colorama.Fore.RED + f"Error in main loop: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        print(colorama.Fore.CYAN + "Application terminated.")

if __name__ == "__main__":
    main() 