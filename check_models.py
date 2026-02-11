import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: No API Key found.")
else:
    genai.configure(api_key=api_key)
    print(f"Checking models for key: {api_key[:5]}...")
    
    try:
        print("\n--- AVAILABLE MODELS ---")
        found = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
                found = True
        
        if not found:
            print("No models found.")
            
    except Exception as e:
        print(f"Error listing models: {e}")