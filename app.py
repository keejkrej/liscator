from flask import Flask, render_template, request, redirect, url_for
from flask.json import jsonify
from gui import CellViewer

app = Flask(__name__)

# Initialize the global variable at the module level
cell_viewer = None

@app.route('/test')
def test():
    return 'test'

@app.route('/')
def index():
    nd2_paths = load_paths('nd2_paths.txt')
    out_paths = load_paths('out_paths.txt')
    if not nd2_paths or not out_paths:
        # Handle the error appropriately, e.g., return an error message or redirect
        return "Error: Missing file paths", 400
    return render_template('index.html', nd2_paths=nd2_paths, out_paths=out_paths)

@app.route('/select_paths', methods=['POST'])
def select_paths():
    global cell_viewer
    nd2_path = request.form['nd2_paths']
    out_path = request.form['out_paths']
    # Use the selected paths as needed
    cell_viewer = CellViewer(nd2_path=nd2_path, output_path=out_path)
    return redirect(url_for('view'))

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     # Request paths for CellViewer
#     # if request.method == 'POST':
#     #     nd2_path = request.form['nd2_path']
#     #     output_path = request.form['output_path']



#     return render_template('index.html')#, channel_image=img, num_channels=3)

# @app.route('/set_paths', methods=['POST'])
# def set_paths():
#     global cell_viewer
#     nd2_path = request.files.getlist('nd2_path')[0].filename
#     out_path = request.files.getlist('out_path')[0].filename
#     cell_viewer = CellViewer(nd2_path=nd2_path, output_path=out_path)
#     return redirect(url_for('view'))

@app.route('/view', methods=['GET', 'POST'])
def view():
    global cell_viewer

    cell_viewer.position_changed()
    return render_template('view.html', channel_image=cell_viewer.return_image(), n_positions=len(cell_viewer.positions), num_channels=cell_viewer.channel_max)#, data=data) #, channel_image=img)

@app.route('/update_image', methods=['POST'])
def update_image():
    global cell_viewer
    # cell_viewer.position_changed()
    new_position = int(request.form['position'])
    new_channel = int(request.form['channel'])

    cell_viewer.channel = new_channel
    cell_viewer.position = cell_viewer.position_options[new_position]
    # cell_viewer.channel = new_position
    cell_viewer.get_channel_image()
    return jsonify({'channel_image': cell_viewer.return_image()})

def load_paths(file_path):
    with open(file_path, mode='r') as file:
        paths = [line.strip() for line in file]
    return paths

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000, debug=True)

