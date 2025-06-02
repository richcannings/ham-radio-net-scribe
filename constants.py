# Constants
DEVICE_NAME = "USB PnP Sound Device: Audio (hw:2,0)" # Target audio device name
WHISPER_MODEL = 'small.en' # As per user's file

# VAD and Audio Recording Parameters
NEW_SILENCE_DURATION_SECONDS = 0.7  # Duration of silence to end recording
SD_SOUND_AMPLITUDE_THRESHOLD = 0.02 # Amplitude threshold for sound detection
FRAME_DURATION_SECONDS = 0.03     # Duration of small frames for VAD

CAPTURE_SAMPLE_RATE = 48000       # Sample rate for capturing audio
CAPTURE_CHANNELS = 1
CAPTURE_DTYPE = 'float32'         # Data type for sounddevice capture

WHISPER_TARGET_SAMPLE_RATE = 16000 # Whisper expects 16kHz