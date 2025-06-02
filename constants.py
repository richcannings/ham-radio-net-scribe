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