import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import synthesize_kilonova_spectrum

print("Starting reproduction script...")
wav_grid_AA = np.linspace(3000, 25000, 100)

print("Synthesizing 1.17d spectrum...")
try:
    flux_1_17d = synthesize_kilonova_spectrum(wav_grid_AA, epoch_days=1.17, T_phot=5000, v_phot=0.2, v_max=0.5, X_Sr=0.001)
    print("1.17d synthesis successful!")
except Exception as e:
    print(f"Error during 1.17d synthesis: {e}")

print("Synthesizing 1.43d spectrum...")
try:
    flux_1_43d = synthesize_kilonova_spectrum(wav_grid_AA, epoch_days=1.43, T_phot=4500, v_phot=0.18, v_max=0.5, X_Sr=0.001)
    print("1.43d synthesis successful!")
except Exception as e:
    print(f"Error during 1.43d synthesis: {e}")

print("Reproduction test complete.")
