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

import google.generativeai as genai
import colorama # For potential debug prints within the class if ever needed

class GeminiManager:
    GEMINI_API_KEY_FILE = 'gemini_api_key.txt'
    GEMINI_MODEL = 'models/gemini-1.5-flash-latest'
    GEMINI_SYSTEM_PROMPT = """You are an expert amateur radio (ham radio) net scribe. Your sole purpose is to listen to a transcription of net check-ins and rapidly extract specific pieces of information.

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
    GEMINI_SAFETY_SETTINGS = [
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

    def __init__(self):
        try:
            with open(self.GEMINI_API_KEY_FILE, 'r') as f:
                gemini_api_key = f.read().strip()
            if not gemini_api_key:
                raise ValueError("Gemini API key is empty in file.")
            genai.configure(api_key=gemini_api_key)
        except FileNotFoundError:
            print(colorama.Fore.RED + f"CRITICAL ERROR: The Gemini API key file '{self.GEMINI_API_KEY_FILE}' was not found.")
            print(colorama.Fore.RED + "Please create this file and add your API key to it. Exiting.")
            exit(1)
        except ValueError as e:
            print(colorama.Fore.RED + f"CRITICAL ERROR: {e}")
            print(colorama.Fore.RED + "Please ensure your API key is correctly placed in the file. Exiting.")
            exit(1)

        self.generation_config = genai.types.GenerationConfig(temperature=0)
        self.model = genai.GenerativeModel(
            model_name=self.GEMINI_MODEL,
            generation_config=self.generation_config,
            system_instruction=self.GEMINI_SYSTEM_PROMPT,
            safety_settings=self.GEMINI_SAFETY_SETTINGS
        )
        print(colorama.Fore.CYAN + "GeminiManager initialized successfully.")

    def get_structured_output(self, transcribed_text: str) -> str | None:
        """Sends transcribed text to Gemini and returns the processed, structured output."""
        if not transcribed_text:
            return None
        
        try:
            gemini_response = self.model.generate_content(transcribed_text)
            gemini_output_text = gemini_response.text.strip()
            processed_gemini_output = gemini_output_text

            json_md_prefix = "```json"
            generic_md_prefix = "```"
            md_suffix = "```"

            if gemini_output_text.startswith(json_md_prefix) and gemini_output_text.endswith(md_suffix):
                processed_gemini_output = gemini_output_text[len(json_md_prefix):-len(md_suffix)].strip()
            elif gemini_output_text.startswith(generic_md_prefix) and gemini_output_text.endswith(md_suffix):
                processed_gemini_output = gemini_output_text[len(generic_md_prefix):-len(md_suffix)].strip()
            
            return processed_gemini_output
        except Exception as e:
            print(colorama.Fore.RED + f"Error during Gemini API call: {e}")
            import traceback
            print(traceback.format_exc())
            return None 