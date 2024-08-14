import gradio as st
import argparse, json
from test import main


def main_loop_strl(descri1, descri2, descri3, g_scale, thoigian, uploaded_file):
    global outmp4list
    descri = list()
    for s in (descri1, descri2, descri3, ):
        if s is not None: descri += [s]
    for out___mp4_, audio_values, descri in main_loop(descri, g_scale, thoigian, "___", False, uploaded_file):
        outmp4 = st.Audio(value=(sampling_rate, audio_values, ), label=descri, visible=True)
        outmp4list.append(outmp4)
    return outmp4list

def showdata_col1():
    descri1 = st.Textbox(label="descri1")
    return descri1

def showdata_col2():
    descri2 = st.Textbox(label="descri2")
    thoigian = st.Slider(
        5, 300,
        step=1.0,
        label="Time to generate music",
        value=8,
    )
    uploaded_file = st.UploadButton("Choose a file", file_types=["audio"])
    return thoigian, descri2, uploaded_file

def showdata_col3():
    descri3 = st.Textbox(label="descri3")
    return descri3


outmp4list = list()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default='8501',
        help='Port to run the server listener on',
    )
    args = parser.parse_args()
    server_port = args.server_port
    server_name = args.listen

    def foo(dir):
        jso = main(dir)
        return json.dumps(jso, sort_keys=True, indent=4)

    with st.Blocks() as demo:
        input = st.File(file_count="directory")
        files = st.Textbox()
        show = st.Button(value="classification")
        show.click(foo, input, files)

    demo.launch(server_port=server_port, server_name=server_name)
