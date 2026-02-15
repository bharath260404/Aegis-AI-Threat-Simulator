import pyttsx3
import threading
import pythoncom
import speech_recognition as sr
import time

# Lock to prevent voice threads from crashing into each other
voice_lock = threading.Lock()

def _speak_worker(text):
    with voice_lock: 
        try:
            pythoncom.CoInitialize()
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            try:
                engine.setProperty('voice', voices[1].id)
            except:
                engine.setProperty('voice', voices[0].id)
            engine.setProperty('rate', 175) 
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Voice Error: {e}")
        finally:
            pythoncom.CoUninitialize()

class AegisVoice:
    def speak(self, text):
        """Spawns a background thread to speak."""
        if text:
            threading.Thread(target=_speak_worker, args=(text,)).start()

    def list_microphones(self):
        """Returns a list of available microphones with their index."""
        try:
            mics = sr.Microphone.list_microphone_names()
            return [(i, name) for i, name in enumerate(mics)]
        except:
            return []

    def listen(self, device_index=None):
        """Listens to the specific microphone index selected in the UI."""
        r = sr.Recognizer()
        
        # SENSITIVITY SETTINGS
        r.energy_threshold = 300  
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.8   
        
        try:
            # Use the specific index passed from the App
            with sr.Microphone(device_index=device_index) as source:
                print(f"🎤 CONNECTED TO DEVICE INDEX: {device_index}")
                
                # Fast calibration (0.5s)
                r.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen
                audio = r.listen(source, timeout=5, phrase_time_limit=8)
                
                text = r.recognize_google(audio)
                return text.lower()
                
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except Exception as e:
            print(f"Mic Error: {e}")
            return None

# Global Instance
ai = AegisVoice()