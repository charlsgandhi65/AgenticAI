
import os
from pathlib import Path
import sysconfig


if os.name == "nt":
    nvidia_dir = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    dll_directories = []
    for package in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
        dll_dir = nvidia_dir / package / "bin"
        if dll_dir.is_dir():
            dll_directories.append(str(dll_dir))
            os.add_dll_directory(str(dll_dir))
    if dll_directories:
        os.environ["PATH"] = os.pathsep.join(dll_directories + [os.environ["PATH"]])

from faster_whisper import WhisperModel

AUDIO_FILE = "test_audio.m4a"

model = WhisperModel(
    "base.en",
    device="cuda",
    compute_type="float16",
)

segments, info = model.transcribe(
    AUDIO_FILE,
    beam_size=5,
)

print(f"Detected language: {info.language}")
print("Transcription:")

for segment in segments:
    print(segment.text)

print("STT test completed.")
