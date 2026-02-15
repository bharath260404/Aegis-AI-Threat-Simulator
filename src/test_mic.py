import speech_recognition as sr

def test_microphone():
    print("------------------------------------------------")
    print("🎤 AUDIO DIAGNOSTIC TOOL")
    print("------------------------------------------------")
    
    r = sr.Recognizer()
    
    # 1. List all devices
    mics = sr.Microphone.list_microphone_names()
    print(f"Found {len(mics)} devices:")
    for i, mic_name in enumerate(mics):
        print(f"   [{i}] {mic_name}")

    print("------------------------------------------------")
    index = int(input("Enter the number [X] of your Headset/Microphone: "))
    
    print(f"\nTesting Device [{index}]...")
    try:
        with sr.Microphone(device_index=index) as source:
            print(">> ADJUSTING FOR NOISE (Stay quiet for 1 sec)...")
            r.adjust_for_ambient_noise(source, duration=1.0)
            print(f">> LISTENING NOW (Speak into mic detected at index {index})...")
            audio = r.listen(source, timeout=5)
            print(">> PROCESSING...")
            
            try:
                text = r.recognize_google(audio)
                print(f"✅ SUCCESS! I heard: '{text}'")
            except sr.UnknownValueError:
                print("❌ ERROR: Audio detected, but could not understand words.")
            except sr.RequestError as e:
                print(f"❌ ERROR: Could not reach Google API. Check internet. {e}")
                
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        print("NOTE: If you see 'PyAudio' errors, you must install it.")

if __name__ == "__main__":
    test_microphone()