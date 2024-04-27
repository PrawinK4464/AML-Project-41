import speech_recognition as sr

recognizer = sr.Recognizer()
audio_file = "harvard.wav"
with sr.AudioFile(audio_file) as source:
  audio_data = recognizer.record(source)

  try:
    text = recognizer.recognize_google(audio_data)
    print("Speech:", text)
  except sr.UnknownValueError:
    print("Not Found")
  except sr.RequestError as e:
    print("could not request result; {0}".format(e))
