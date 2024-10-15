from flask import Flask, render_template, request, redirect, url_for, jsonify
from gui import CellViewer
import pyama_util
import os

class App:
    def __init__(self):
        self.app = Flask(__name__)
        self.cell_viewer = None

    def routes(self):
        @self.app.route('/test')
        def test():
            return 'test'

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/select_paths', methods=['POST'])
        def select_paths():
            data = request.json
            nd2_path = data['nd2_path']
            out_path = data['out_path']
            redirect_to = data['redirect_to']

            if not nd2_path or not out_path:
                return jsonify({'error': 'Both ND2 path and output path must be selected'}), 400

            self.cell_viewer = CellViewer(nd2_path=nd2_path, output_path=out_path)
            self.cell_viewer.nd2_path = nd2_path
            self.cell_viewer.output_path = out_path

            if redirect_to == 'view':
                return jsonify({'redirect': url_for('view')})
            elif redirect_to == 'analysis':
                return jsonify({'redirect': url_for('analysis')})
            else:
                return jsonify({'redirect': url_for('index')})

        @self.app.route('/view', methods=['GET', 'POST'])
        def view():
            if self.cell_viewer is None:
                return redirect(url_for('index'))
            self.cell_viewer.position_changed()
            current_particle_index = self.cell_viewer.all_particles.index(self.cell_viewer.particle)
            return render_template('view.html',
                                   channel_image=self.cell_viewer.return_image(),
                                   n_positions=len(self.cell_viewer.positions),
                                   n_channels=self.cell_viewer.channel_max,
                                   n_frames=self.cell_viewer.frame_max,
                                   all_particles_len=self.cell_viewer.all_particles_len,
                                   current_particle_index=current_particle_index,
                                   brightness_plot=self.cell_viewer.brightness_plot)

        @self.app.route('/preprocess', methods=['GET', 'POST'])
        def processing():
            return render_template('preprocess.html')

        @self.app.route('/documentation', methods=['GET', 'POST'])
        def documentation():
            svg = "static/images/UserTutorial.svg"
            return render_template('documentation.html', svg=svg)

        @self.app.route('/update_image', methods=['POST'])
        def update_image():
            new_position = int(request.form['position'])
            new_channel = int(request.form['channel'])
            new_frame = int(request.form['frame'])
            new_particle = int(request.form['particle'])

            if new_position != self.cell_viewer.position:
                self.cell_viewer.position = self.cell_viewer.position_options[new_position]
                self.cell_viewer.position_changed()

            if new_particle != self.cell_viewer.particle:
                self.cell_viewer.particle = self.cell_viewer.all_particles[new_particle]
                self.cell_viewer.particle_changed()

            self.cell_viewer.channel = new_channel
            self.cell_viewer.frame = new_frame

            self.cell_viewer.get_channel_image()
            self.cell_viewer.draw_outlines()
            return jsonify({'channel_image': self.cell_viewer.return_image(),
                            'brightness_plot': self.cell_viewer.brightness_plot})

        @self.app.route('/do_segmentation', methods=['POST'])
        def do_segmentation():
            data = request.json
            nd2_path = self.cell_viewer.nd2_path
            out_dir = self.cell_viewer.output_path
            positions = list(range(data['position_min'], data['position_max'] + 1))
            frame_min = data['frame_min']
            frame_max = data['frame_max']

            segmentation_channel = []
            fluorescence_channels = []
            for i in range(self.cell_viewer.channel_max + 1):
                channel_type = data[f'channel_{i}']
                if channel_type == 'Brightfield':
                    segmentation_channel.append(i)
                elif channel_type == 'Fluorescent':
                    fluorescence_channels.append(i)

            segmentation_channel = segmentation_channel[0] if len(segmentation_channel) == 1 else segmentation_channel

            pyama_util.segment_positions(nd2_path, out_dir, positions, segmentation_channel, fluorescence_channels, frame_min=frame_min, frame_max=frame_max)
            return jsonify({'status': 'success'})

        @self.app.route('/do_tracking', methods=['POST'])
        def do_tracking():
            data = request.json
            out_dir = self.cell_viewer.output_path
            positions = list(range(data['position_min'], data['position_max'] + 1))
            expand_labels = data['expand_labels']

            pyama_util.tracking_pyama(out_dir, positions, expand=expand_labels)
            return jsonify({'status': 'success'})

        @self.app.route('/do_square_rois', methods=['POST'])
        def do_square_rois():
            data = request.json
            out_dir = self.cell_viewer.output_path
            positions = list(range(data['position_min'], data['position_max'] + 1))
            square_um_size = data['square_size']

            pyama_util.square_roi(out_dir, positions, square_um_size)
            return jsonify({'status': 'success'})

        @self.app.route('/analysis')
        def analysis():
            if self.cell_viewer is None:
                return redirect(url_for('index'))
            return render_template('analysis.html',
                                   n_positions=len(self.cell_viewer.positions),
                                   n_channels=self.cell_viewer.channel_max,
                                   n_frames=self.cell_viewer.frame_max)

        @self.app.route('/list_directory', methods=['GET'])
        def list_directory():
            path = request.args.get('path', '/')
            try:
                items = os.listdir(path)
                return jsonify({
                    'path': path,
                    'items': [{'name': item, 'isDirectory': os.path.isdir(os.path.join(path, item))} for item in items]
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 400

        @self.app.route('/select_folder', methods=['POST'])
        def select_folder():
            path = request.json['path']
            # Here you can add logic to handle the selected folder
            return jsonify({'message': f'Folder selected: {path}'})

    def load_paths(self, file_path):
        with open(file_path, mode='r') as file:
            paths = [line.strip() for line in file]
        return paths

    def run(self):
        self.app.run(host='0.0.0.0', port=8000, debug=True)

app_instance = App()
app_instance.routes()
flask_app = app_instance.app

if __name__ == '__main__':
    app_instance.run()
