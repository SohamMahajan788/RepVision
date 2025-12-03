import os
import urllib.request
import time

# Create sounds directory if it doesn't exist
os.makedirs('sounds', exist_ok=True)

# Sample URLs for sound effects
# In a real application, you would replace these with actual sound files
sound_urls = {
    'good_rep': 'https://www.soundjay.com/button/sounds/button-09.mp3',
    'bad_form': 'https://www.soundjay.com/button/sounds/button-10.mp3',
    'good_form': 'https://www.soundjay.com/button/sounds/button-1.mp3'
}

print("Downloading sound files...")

for sound_name, url in sound_urls.items():
    try:
        filepath = f'sounds/{sound_name}.mp3'
        print(f"Downloading {sound_name} sound...")
        
        # Download the file
        urllib.request.urlretrieve(url, filepath)
        
        print(f"Downloaded {sound_name} sound to {filepath}")
        
        # Small delay to avoid overwhelming the server
        time.sleep(1)
    except Exception as e:
        print(f"Error downloading {sound_name} sound: {e}")
        
        # Create an empty file as a placeholder
        with open(f'sounds/{sound_name}.mp3', 'w') as f:
            pass
        
        print(f"Created empty placeholder for {sound_name}")

print("Sound download complete!") 