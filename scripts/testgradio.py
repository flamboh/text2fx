import gradio_client.utils as gu

# --- Full monkey patch to make Gradio tolerant of boolean schemas ---
_old_json_schema_to_python_type = gu._json_schema_to_python_type
_old_get_type = gu.get_type

def safe_json_schema_to_python_type(schema, defs=None):
    # If schema is True/False, return a dummy string type
    if isinstance(schema, bool):
        return "object" if schema else "never"
    try:
        return _old_json_schema_to_python_type(schema, defs)
    except Exception:
        return "object"

def safe_get_type(schema):
    if isinstance(schema, bool):
        return "object" if schema else "never"
    try:
        return _old_get_type(schema)
    except Exception:
        return "object"

gu._json_schema_to_python_type = safe_json_schema_to_python_type
gu.get_type = safe_get_type
# --- End of patch ---
import gradio as gr
import numpy as np

def explore(x, y):
    sr = 44100
    t = np.linspace(0, 1, sr)
    freq = 220 + 220 * x + 100 * y
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return (sr, audio)

gr.Interface(
    fn=explore,
    inputs=[
        gr.Slider(-1, 1, step=0.1, label="X Axis: Dark ↔ Bright"),
        gr.Slider(-1, 1, step=0.1, label="Y Axis: Wooden ↔ Metallic"),
    ],
    outputs=gr.Audio(label="Dummy Audio Output"),
    live=True,
).launch(server_port=7869, share=True)
