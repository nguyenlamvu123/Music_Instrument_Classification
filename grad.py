import gradio as st
import argparse, json, joblib
from test import main, Model


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
    clf = joblib.load(Model.NAME)

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
        jso = main(dir, clf=clf)
        return json.dumps(jso, sort_keys=True, indent=4)

    with st.Blocks() as demo:
        input = st.File(file_count="directory")
        files = st.Textbox()
        show = st.Button(value="classification")
        show.click(foo, input, files)

    demo.launch(server_port=server_port, server_name=server_name)
