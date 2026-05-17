
import speech_recognition as sr
from langgraph.checkpoint.mongodb import MongoDBSaver
import os
from gtts import gTTS
from graph import create_chat_graph
import pyttsx3
MONGODB_URL = "mongodb://admin:admin@localhost:27017"
config = {
        "configurable":{"thread_id":"15"},
          "recursion_limit":6
           }


def main():
    with MongoDBSaver.from_conn_string(MONGODB_URL) as checkpointer:
        graph_with_mongo = create_chat_graph(checkpointer=checkpointer)
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.pause_threshold = 2
            while True:
                print("Say something!")
                audio = r.listen(source)
                
                print("Processing audio...")
                sst = r.recognize_google(audio)
                print("You said:",sst)
            
                for events in graph_with_mongo.stream({"messages":[{"role":"user","content":sst}]},config,stream_mode="values"):
                    if "messages" in events:
                        last_msg = events["messages"][-1]
                        last_msg.pretty_print()
                    if last_msg.type == "ai":
                        if isinstance(last_msg.content, list):
                            parts = []

                            for item in last_msg.content:
                                if isinstance(item,dict):
                                    if item.get("type") == "text":
                                        parts.append(item.get("text",""))
                                elif isinstance(item,str):
                                    parts.append(item)
                            text = " ".join(parts)
                        else:
                            text = str(last_msg.content)

                        speak(text)

engine = pyttsx3.init()  # initialize once (important!)
engine.setProperty('rate', 160)   # speed (default ~200)
engine.setProperty('volume', 1.0)

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # try 0 / 1
def speak(text):
    if not isinstance(text, str):
        text = str(text)

    engine.say(text)
    engine.runAndWait()

main()