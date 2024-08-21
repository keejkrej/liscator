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

@app.route('/set_paths', methods=['POST'])
def set_paths():
    global cell_viewer
    nd2_path = request.files.getlist('nd2_path')[0].filename
    out_path = request.files.getlist('out_path')[0].filename
    cell_viewer = CellViewer(nd2_path=nd2_path, output_path=out_path)
    return redirect(url_for('view'))

@app.route('/view', methods=['GET', 'POST'])
def view():
    # return ('test')
    global cell_viewer
    # if cell_viewer is None:
    #     cell_viewer = CellViewer(nd2_path='path/to/nd2', output_path='path/to/output')
    
    # # Example of changing an attribute
    # cell_viewer.set_position('new_position')
    
    # data = cell_viewer.render_image()  # Replace with actual method
    # return('test')
    cell_viewer.position= ['XY00', 'XY01', 'XY02']
    cell_viewer.position_changed()
    return render_template('view.html', channel_image=cell_viewer.return_image())#, data=data) #, channel_image=img, num_channels=3)

def load_paths(file_path):
    with open(file_path, mode='r') as file:
        paths = [line.strip() for line in file]
    return paths

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000, debug=True)

