import numpy as np
import whisper
import google.generativeai as genai
import colorama
import threading
import sounddevice as sd # New import
from scipy.signal import resample # New import
# scipy.io.wavfile is no longer needed as debug audio saving is removed for now

# Constants
DEVICE_NAME = "USB PnP Sound Device: Audio (hw:2,0)" # Target audio device name
GEMINI_API_KEY_FILE = 'gemini_api_key.txt'
# User updated GEMINI_MODEL and WHISPER_MODEL, and Gemini prompt formatting
GEMINI_MODEL = 'models/gemini-1.5-flash-latest' # As per user's file
WHISPER_MODEL = 'small.en' # As per user's file

# VAD and Audio Recording Parameters (New/Updated based on sounddevice approach)
NEW_SILENCE_DURATION_SECONDS = 0.7  # Duration of silence to end recording
SD_SOUND_AMPLITUDE_THRESHOLD = 0.02 # Amplitude threshold for sound detection with sounddevice (float32 range [-1,1])
FRAME_DURATION_SECONDS = 0.03     # Duration of small frames for VAD (e.g., 30ms)

CAPTURE_SAMPLE_RATE = 48000       # Sample rate for capturing audio from device (e.g., 48000 or 44100)
CAPTURE_CHANNELS = 1
CAPTURE_DTYPE = 'float32'         # Data type for sounddevice capture

WHISPER_TARGET_SAMPLE_RATE = 16000 # Whisper expects 16kHz

# Gemini and Whisper Initialization
# Whisper Model
# Ensure WHISPER_MODEL is used from constants
whisper_model = whisper.load_model(WHISPER_MODEL)

# Gemini API
try:
    with open(GEMINI_API_KEY_FILE, 'r') as f:
        gemini_api_key = f.read().strip()
    if not gemini_api_key:
        raise ValueError("Gemini API key is empty.")
    genai.configure(api_key=gemini_api_key)
except FileNotFoundError:
    print(f"Error: The Gemini API key file '{GEMINI_API_KEY_FILE}' was not found.")
    print("Please create this file and add your API key to it.")
    exit()
except ValueError as e:
    print(f"Error: {e}")
    print("Please ensure your API key is correctly placed in the file.")
    exit()

# Instantiate the Gemini model and define the system prompt
gemini_generation_config = genai.types.GenerationConfig(temperature=0) # For deterministic output
gemini_safety_settings = [ # Adjust safety settings as needed
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

# Updated Gemini System Prompt based on user's local changes
gemini_system_prompt = """You are an expert amateur radio (ham radio) net scribe. Your sole purpose is to listen to a transcription of net check-ins and rapidly extract specific pieces of information.

From the transcription, identify the following four items:
1.  **Call Sign:** The operator's official call sign (e.g., W1AW, K2MGA). It may be given phonetically (e.g., "November Six Tango Victor").
2.  **Name:** The operator's first name.
3.  **Location:** The operator's general location (e.g., "Santa Cruz," "downtown").
4.  **Traffic:** Whether the operator has "traffic" (a message for the net). This is usually indicated by the words "traffic," "no traffic," or "nothing for the net." The answer should be "true" or "false".

Your output MUST be a JSON string, formatted exactly as follows:
{
  "callsign": "[The call sign]",
  "name": "[The name]",
  "location": "[The location]",
  "traffic": [true/false]
}

If a piece of information is not present, use "unknown" for its value."""

gemini_model_instance = genai.GenerativeModel(
    model_name=GEMINI_MODEL, # Uses the GEMINI_MODEL constant from above
    generation_config=gemini_generation_config,
    system_instruction=gemini_system_prompt,
    safety_settings=gemini_safety_settings
)

# Audio Processing Thread (Revised for sounddevice)
def process_audio_thread(raw_audio_data, capture_samplerate):
    """Processes recorded audio (from sounddevice) in a separate thread."""
    if raw_audio_data is None or raw_audio_data.size == 0:
        print(colorama.Fore.CYAN + "No audio data received for processing.")
        return

    try:
        # Ensure audio is float32, which it should be from sounddevice with CAPTURE_DTYPE='float32'
        if raw_audio_data.dtype != np.float32:
            print(colorama.Fore.YELLOW + f"Warning: Audio data is {raw_audio_data.dtype}, converting to float32.")
            processed_audio = raw_audio_data.astype(np.float32)
        else:
            processed_audio = raw_audio_data

        # Resample if necessary
        if capture_samplerate != WHISPER_TARGET_SAMPLE_RATE:
            num_samples = int(len(processed_audio) * WHISPER_TARGET_SAMPLE_RATE / capture_samplerate)
            processed_audio = resample(processed_audio, num_samples)
            current_samplerate = WHISPER_TARGET_SAMPLE_RATE
            print(colorama.Fore.MAGENTA + f"Resampled audio from {capture_samplerate} Hz to {current_samplerate} Hz.")
        else:
            current_samplerate = capture_samplerate

        # Heuristic: if audio is less than 1.0s (user changed from 0.1s in their local copy)
        # This check should now use the resampled audio length and WHISPER_TARGET_SAMPLE_RATE
        min_duration_for_transcription = 1.0 # seconds
        if len(processed_audio) < WHISPER_TARGET_SAMPLE_RATE * min_duration_for_transcription:
            print(colorama.Fore.CYAN + f"Audio too short to transcribe meaningfully (less than {min_duration_for_transcription}s after resampling).")
            return

        # Log audio characteristics before sending to Whisper
        print(colorama.Fore.MAGENTA +
              f"Audio for Whisper - Dtype: {processed_audio.dtype}, Shape: {processed_audio.shape}, Sample Rate: {current_samplerate}, " +
              f"Min: {np.min(processed_audio):.4f}, Max: {np.max(processed_audio):.4f}, Mean: {np.mean(processed_audio):.4f}")

        # Transcribe with Whisper
        # User's example used fp16=True
        transcript = whisper_model.transcribe(processed_audio, fp16=True, language='en')

        transcribed_text = transcript['text'].strip()
        print(colorama.Fore.YELLOW + "Transcription: " + transcribed_text)

        # Send to Gemini
        if transcribed_text:
            # Gemini expects a string. The prompt asks for JSON output.
            gemini_response = gemini_model_instance.generate_content(transcribed_text)
            
            gemini_output_text = gemini_response.text.strip() # Start with a clean, stripped string
            processed_gemini_output = gemini_output_text # Default to the original stripped text

            # Define potential prefixes and suffix
            json_md_prefix = "```json"
            generic_md_prefix = "```"
            md_suffix = "```"

            # More robustly strip markdown code blocks
            if gemini_output_text.startswith(json_md_prefix) and gemini_output_text.endswith(md_suffix):
                # Slice off the prefix and suffix, then strip any leading/trailing whitespace from the content itself.
                processed_gemini_output = gemini_output_text[len(json_md_prefix):-len(md_suffix)].strip()
            elif gemini_output_text.startswith(generic_md_prefix) and gemini_output_text.endswith(md_suffix):
                # Slice off the prefix and suffix, then strip any leading/trailing whitespace from the content itself.
                processed_gemini_output = gemini_output_text[len(generic_md_prefix):-len(md_suffix)].strip()

            print(colorama.Fore.GREEN + "Gemini Output:")
            # Print the processed output, which might be the original or the stripped version
            print(colorama.Fore.GREEN + processed_gemini_output)
        else:
            print(colorama.Fore.CYAN + "No text from transcription to send to Gemini.")

    except Exception as e:
        print(colorama.Fore.RED + f"Error in processing thread: {e}")
        import traceback
        print(traceback.format_exc()) # Print full traceback for debugging

# Main Listening Loop (Revised for sounddevice)

def find_audio_device_index_sd(device_name_query):
    """Finds the audio device index for sounddevice based on a partial name match."""
    devices = sd.query_devices()
    print("Available audio devices:") # Simplified print
    matching_device_index = None
    for i, dev_info in enumerate(devices):
        # Quieter listing, only print essential info
        print(f"  {i}: {dev_info['name']} (In: {dev_info['max_input_channels']}, Rate: {dev_info['default_samplerate']})") 
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
                # print(f"Amplitude: {amplitude:.4f}", end='\r') # Debug VAD

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