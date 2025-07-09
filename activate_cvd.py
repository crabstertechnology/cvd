import pyttsx3

engine = pyttsx3.init()

# Set female voice
voices = engine.getProperty('voices')
for voice in voices:
    if 'Zira' in voice.name:
        engine.setProperty('voice', voice.id)

engine.say("Welcome Boss! Your Environment is activated")
engine.runAndWait()
