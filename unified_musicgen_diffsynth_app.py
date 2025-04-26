import sys
import os
import gradio as gr

# Add paths for local imports
AUDIOCRAFT_PATH = os.path.abspath(os.path.dirname(__file__))
DIFFSYNTH_PATH = r'C:\Users\Administrator\DiffSynth-Studio'
sys.path.append(AUDIOCRAFT_PATH)
sys.path.append(DIFFSYNTH_PATH)

# Import MusicGen (AudioCraft)
try:
    from audiocraft.models import MusicGen
except ImportError:
    MusicGen = None

# Import DiffSynth (assume main interface is in diffsynth)
try:
    from diffsynth.pipelines import AudioPipeline
except ImportError:
    AudioPipeline = None

def generate_musicgen(prompt, duration):
    try:
        if MusicGen is None:
            return None, "MusicGen not available. Check installation."
        print(f"[DEBUG] Calling MusicGen with prompt='{prompt}', duration={duration}")
        model = MusicGen.get_pretrained('facebook/musicgen-melody')
        wav = model.generate([prompt], progress=True, duration=duration)
        import tempfile, soundfile as sf
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_wav.name, wav[0].cpu().numpy().T, 32000)
        print(f"[DEBUG] MusicGen generated file: {temp_wav.name}")
        return temp_wav.name, None
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] MusicGen failure: {e}\n{tb}")
        return None, f"MusicGen error: {e}\n{tb}"

def generate_diffsynth(prompt, duration):
    try:
        if AudioPipeline is None:
            return None, "DiffSynth not available. Check installation."
        print(f"[DEBUG] Calling DiffSynth with prompt='{prompt}', duration={duration}")
        pipeline = AudioPipeline.load_pretrained()  # or custom loading if needed
        wav, sr = pipeline(prompt=prompt, duration=duration)
        import tempfile, soundfile as sf
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_wav.name, wav, sr)
        print(f"[DEBUG] DiffSynth generated file: {temp_wav.name}")
        return temp_wav.name, None
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] DiffSynth failure: {e}\n{tb}")
        return None, f"DiffSynth error: {e}\n{tb}"

def generate_audio(model_choice, prompt, duration):
    if model_choice == "MusicGen":
        return generate_musicgen(prompt, duration)
    elif model_choice == "DiffSynth":
        return generate_diffsynth(prompt, duration)
    else:
        return None, "Unknown model."

def gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Unified MusicGen & DiffSynth Studio\nGenerate music/audio with your choice of model!")
        model_choice = gr.Radio(["MusicGen", "DiffSynth"], label="Model", value="MusicGen")
        prompt = gr.Textbox(label="Prompt", lines=2)
        duration = gr.Number(label="Duration (seconds)", value=10, precision=2)
        output = gr.Audio(label="Generated Audio")
        error_box = gr.Textbox(label="Error", visible=False)
        def run(model_choice, prompt, duration):
            try:
                # Validate duration
                if duration is None or float(duration) <= 0:
                    return None, "Please enter a positive duration.", gr.update(visible=True)
                audio_path, error = generate_audio(model_choice, prompt, float(duration))
                if error:
                    return None, error, gr.update(visible=True)
                return audio_path, "", gr.update(visible=False)
            except Exception as e:
                return None, f"Error: {e}", gr.update(visible=True)
        btn = gr.Button("Generate")
        btn.click(run, [model_choice, prompt, duration], [output, error_box, error_box])
    return demo

if __name__ == "__main__":
    gradio_ui().launch()
