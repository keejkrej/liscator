import os
import re
import cv2
import h5py
import numpy as np
import pandas as pd
import bqplot.pyplot as bqplt
from nd2reader import ND2Reader

import ipywidgets as widgets
from ipyevents import Event

import pathlib
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class CellViewer:

    def __init__(self, nd2_path, output_path):

        self.output_path = pathlib.Path(output_path)

        self.output_path = output_path
        self.nd2 = ND2Reader(nd2_path)
        self.file = None

        self.COLOR_GRAY = '#808080'
        self.COLOR_RED = 'Red'
        self.COLOR_ORANGE = '#FF8C00'

        self.OPACITY_DEFAULT = 0.5
        self.OPACITY_SELECTED = 1


        self.frame_change_suppress = False

        self.particle = None
        self.position = None

        # parse valid positions
        self.get_positions()
        #print(self.positions)

        self.frame_min = 0
        self.frame_max = self.nd2.metadata['num_frames']-1
        self.frame = self.frame_min

        self.channel = 0
        self.channel_min = 0
        self.channel_max = len(self.nd2.metadata['channels'])-1

        #self.max_pixel_value = np.iinfo(np.uint16).max
        self.max_pixel_value = 10000

        self.image_size = 400
        self.outline_kernel = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,0,1,1],[0,1,1,1,0],[0,0,1,0,0]])

    def get_positions(self):
        self.positions = []
        folders = self.get_subdirs(self.output_path)
        for folder in folders:
            match = re.search(r'^XY0*(\d+)$', folder)
            if not match:
                continue

            pos_files = self.get_files(os.path.join(self.output_path,folder))
            if not 'data.h5' in pos_files:
                continue
            if not 'features.csv' in pos_files:
                continue
            if not 'tracks.csv' in pos_files:
                continue
            #print(pos_files)
            pos = (int(match.group(1)), folder)
            self.positions.append(pos)

        self.positions = sorted(self.positions, key=lambda p: p[0], reverse=False)

    def get_subdirs(self, directory):
        return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory,d))]

    def get_files(self, directory):
        return [d for d in os.listdir(directory) if os.path.isfile(os.path.join(directory,d))]

    def get_track_data(self, particle, field):
        t = self.all_tracks[self.all_tracks['particle'] == particle]
        return t['frame'].values, t[field].values

    def update_plots(self):
        particle_index = self.particle_index()
        disabled_particles = self.all_tracks[self.all_tracks['enabled'] == False]['particle'].unique()

        print("disabled:", disabled_particles)

        # boolean list [True, False] => enabled, disabled
        particle_states = list(self.all_tracks.groupby(['particle'])['enabled'].all())

        # all enabled areas except selected particle
        area_x = [self.area_x[i] for i in range(self.all_particles_len) if particle_states[i] == True and i != particle_index]
        area_y = [self.area_y[i] for i in range(self.all_particles_len) if particle_states[i] == True and i != particle_index]

        # selected area
        area_x.append(self.area_x[particle_index])
        area_y.append(self.area_y[particle_index])

        # equivalent for brightness
        brightness_x = [self.brightness_x[i] for i in range(self.all_particles_len) if particle_states[i] == True and i != particle_index]
        brightness_y = [self.brightness_y[i] for i in range(self.all_particles_len) if particle_states[i] == True and i != particle_index]

        # selected area
        brightness_x.append(self.brightness_x[particle_index])
        brightness_y.append(self.brightness_y[particle_index])

        opacities = [self.OPACITY_DEFAULT] * len(area_x)
        opacities[len(opacities)-1] = self.OPACITY_SELECTED

        colors = [self.COLOR_GRAY] * len(area_x)
        if self.particle_enabled == True:
            colors[len(colors)-1] = self.COLOR_RED
        else:
            colors[len(colors)-1] = self.COLOR_ORANGE

        # Update brightness tracks
        with self.brightness_lines.hold_sync():
            self.brightness_lines.x = brightness_x
            self.brightness_lines.y = brightness_y
            self.brightness_lines.opacities = opacities
            self.brightness_lines.colors = colors

        # Update area tracks
        with self.area_lines.hold_sync():
            self.area_lines.x = area_x
            self.area_lines.y = area_y
            self.area_lines.opacities = opacities
            self.area_lines.colors = colors


    def position_changed(self):
        self.data_dir = os.path.join(self.output_path,self.position[1])
        if self.file is not None:
            self.file.close()
        self.file = h5py.File(os.path.join(self.data_dir,'data.h5'), "r")
        self.frame_min = self.file.attrs['frame_min']
        self.frame_max = self.file.attrs['frame_max']
        self.frame = self.frame_min

        self.brightness_figure.title = self.file.attrs['fl_channel_names'][0]

        # set Brightnesses names for plots file_handle.attrs['fl_channel_names']

        self.all_tracks = pd.read_csv(os.path.join(self.data_dir,'tracks.csv'))
        self.all_particles = list(self.all_tracks['particle'].unique())
        self.all_particles_len = len(self.all_particles)

        self.brightness_x = []
        self.brightness_y = []
        for p in self.all_particles:
            x,y = self.get_track_data(p, 'brightness_0')
            self.brightness_x.append(x)
            self.brightness_y.append(y)

        self.area_x = []
        self.area_y = []
        for p in self.all_particles:
            x,y = self.get_track_data(p, 'area')
            self.area_x.append(x)
            self.area_y.append(y)

        colors = [self.COLOR_GRAY] * len(self.all_particles)
        colors[len(self.all_particles)-1] = self.COLOR_RED

        opacities = [self.OPACITY_DEFAULT] * len(self.all_particles)
        opacities[len(self.all_particles)-1] = self.OPACITY_SELECTED

        self.update_cursors()

        with self.brightness_lines.hold_sync():
            self.brightness_lines.colors = colors
            self.brightness_lines.opacities = opacities
            self.brightness_lines.stroke_width = 3

        with self.area_lines.hold_sync():
            self.area_lines.colors = colors
            self.area_lines.opacities = opacities
            self.area_lines.stroke_width = 3


        self.particle = None

        dropdown_options = []
        for particle in self.all_particles:
            dropdown_options.append((str(particle), particle))
        self.particle_dropdown.options = dropdown_options
        self.particle_dropdown.description = 'Cell (' + str(len(self.all_particles)) + '): '

        # Stop slider from updating on every change & edit slider values
        self.frame_change_suppress = True
        if self.frame_min > self.frame_slider.max:
            self.frame_slider.max = self.frame_max
            self.frame_slider.min = self.frame_min
        else:
            self.frame_slider.min = self.frame_min
            self.frame_slider.max = self.frame_max
        self.frame_slider.value = self.frame
        self.frame_change_suppress = False

        # Will be called if position actually changed (not initial)
        if self.particle is None:
            self.particle = self.particle_dropdown.value
            self.particle_changed()
            #self.update_cursors()
        else:
            self.frame_changed()

    # enable / disable current particle and save tracks to file
    def particle_enabled_changed(self):
        self.all_tracks.loc[self.all_tracks['particle'] == self.particle, 'enabled'] = self.particle_enabled
        self.all_tracks.to_csv(self.data_dir + '/tracks.csv')
        self.update_plots()
        self.draw_outlines()
        self.update_image()

    def particle_index(self):
        return self.all_particles.index(self.particle)

    def particle_changed(self):
        enabled = len(self.all_tracks[(self.all_tracks['particle'] == self.particle) & ((self.all_tracks['enabled'] == True))]) > 0

        # set both so no update to file is applied
        self.particle_enabled = enabled
        self.enabled_checkbox.value = enabled

        self.update_plots()

        self.particle_tracks = self.all_tracks[self.all_tracks['particle'] == self.particle]

        # Get new Position for image
        self.x = int(self.particle_tracks['x'].values.mean()) - self.image_size
        self.y = int(self.particle_tracks['y'].values.mean()) - self.image_size

        self.x = max(0,min(self.nd2.metadata['height'] - 2*self.image_size, self.x))
        self.y = max(0,min(self.nd2.metadata['width'] - 2*self.image_size, self.y))

        self.get_channel_image()
        self.draw_outlines()
        self.update_image()

    def particle_dropdown_changed(self, change):
        if change.new is not self.particle:
            self.particle = change.new
            self.particle_changed()

    def position_dropdown_changed(self, change):
        if change.new is not self.position:
            self.position = change.new
            self.position_changed()

    def frame_slider_changed(self, change):
        if self.frame_change_suppress:
            return
        if change.new is not self.frame:
            self.frame = change.new
            self.frame_changed()

    def max_pixel_slider_changed(self, change):
        if change.new is not self.frame:
            self.max_pixel_value = change.new
            self.max_pixel_value_changed()

    def update_cursors(self):
        # Move Brightness Cursor
        with self.brightness_cursor_line.hold_sync():
            self.brightness_cursor_line.x = [self.frame,self.frame]
            self.brightness_cursor_line.y = [0,1]

        # Move Area Cursor
        with self.area_cursor_line.hold_sync():
            self.area_cursor_line.x = [self.frame,self.frame]
            self.area_cursor_line.y = [0,1]

    def frame_changed(self):
        self.update_cursors()
        self.get_channel_image()
        self.draw_outlines()
        self.update_image()

    def channel_changed(self):
        self.get_channel_image()
        self.update_image()

    def max_pixel_value_changed(self):
        self.get_channel_image()
        self.update_image()

    def channel_slider_changed(self, change):
        if change.new is not self.channel:
            self.channel = change.new
            self.channel_changed()

    def enabled_checkbox_changed(self, change):
        if change.new is not self.particle_enabled:
            self.particle_enabled = change.new
            self.particle_enabled_changed()

    def adjust_image(self, image, max_value):
        img = image.copy().astype(np.float64)
        img[img >= max_value] = np.iinfo(np.uint16).max
        img[img < max_value] /= max_value
        img[img < max_value] *= np.iinfo(np.uint16).max
        return img

    def get_channel_image(self):
        img = self.nd2.get_frame_2D(v=self.position[0],c=self.channel,t=self.frame)[self.x:self.x+2*self.image_size,self.y:self.y+2*self.image_size]
        pixel_val = self.max_pixel_value
        if self.channel == 0:
            pixel_val = 40000
        adjusted = self.adjust_image(img,pixel_val)
        self.channel_image = cv2.cvtColor(cv2.convertScaleAbs(adjusted, alpha=1./256., beta=-.49999),cv2.COLOR_GRAY2RGB)

    def update_image(self):
        img = self.combine_images(self.outline_image,self.channel_image,self.outline_mask)
        _, img_enc = cv2.imencode('.jpg', img)
        self.image.value = img_enc.tobytes()

    def get_outline(self, img):
        f64_img = img.astype(np.float64)
        filter_img = cv2.filter2D(src=f64_img, ddepth=-1,kernel=self.outline_kernel) / self.outline_kernel.sum()
        filter_img[filter_img == f64_img] = 0

        mask = (f64_img != filter_img) & (filter_img > 0)
        filter_img[mask] = img[mask]

        return filter_img.astype(img.dtype)

    def combine_images(self,a,b,m):
        mask = cv2.cvtColor(m,cv2.COLOR_GRAY2RGB)
        inv_mask = cv2.bitwise_not(mask)

        ma = cv2.bitwise_and(a,mask)
        mb = cv2.bitwise_and(b,inv_mask)

        return cv2.add(ma,mb)

    def get_particle_label(self):
        tracks = self.all_tracks[(self.all_tracks['frame'] == self.frame) & (self.all_tracks['particle'] == self.particle)]
        if len(tracks) == 0:
            return None
        return int(tracks.iloc[0]['label'])

    def draw_outlines(self):
        if self.frame < self.frame_min or self.frame > self.frame_max:
            self.outline_mask = np.zeros((self.image_size*2,self.image_size*2),dtype=np.uint8)
            self.outline_image = np.zeros((self.image_size*2,self.image_size*2,3),dtype=np.uint8)
            return
        all_labels = self.file['labels'][self.frame-self.frame_min] [self.x:self.x+2*self.image_size,self.y:self.y+2*self.image_size]

        outlines = self.get_outline(all_labels)

        image_shape = (self.channel_image.shape[0],self.channel_image.shape[1],3)
        overlay = np.zeros(image_shape,dtype=np.uint8)

        o = np.zeros(image_shape,dtype=np.uint8)

        # non tracked
        #o = cv2.rectangle(o, (0,0), (image_shape[0],image_shape[1]), (0,0,255), -1)
        #m1 = (outlines != 0).astype(np.uint8)*255
        #overlay = self.combine_images(o,overlay,m1)

        #frame_tracks = self.all_tracks[self.all_tracks['frame'] == self.frame]
        #tracked_labels = frame_tracks['label'].unique()

        # other tracked
        #o = cv2.rectangle(o, (0,0), (image_shape[0],image_shape[1]), (0,255,255), -1)
        #m2 = np.isin(outlines, tracked_labels).astype(np.uint8)*255
        #overlay = self.combine_images(o,overlay,m2)

        frame_tracks = self.all_tracks[self.all_tracks['frame'] == self.frame]
        enabled_labels = frame_tracks[frame_tracks['enabled'] == True]['label'].unique()
        tracked_labels = frame_tracks['label'].unique()

        # all tracked cells
        o = cv2.rectangle(o, (0,0), (image_shape[0],image_shape[1]), (255,0,0), -1)
        m1 = np.isin(outlines, tracked_labels).astype(np.uint8)*255
        overlay = self.combine_images(o,overlay,m1)

        # enabled cells
        o = cv2.rectangle(o, (0,0), (image_shape[0],image_shape[1]), (0,255,0), -1)
        m2 = np.isin(outlines, enabled_labels).astype(np.uint8)*255
        overlay = self.combine_images(o,overlay,m2)

        # Selected cell
        label = self.get_particle_label()
        if label is not None:
            if self.particle_enabled == True:
                o = cv2.rectangle(o, (0,0), (image_shape[0],image_shape[1]), (0,0,255), -1)
            else:
                o = cv2.rectangle(o, (0,0), (image_shape[0],image_shape[1]), (0,140,255), -1)
            m3 = (outlines == label).astype(np.uint8)*255

            overlay = self.combine_images(o,overlay,m3)

        self.outline_image = overlay #self.combine_images(overlay,self.image_data,m1)
        self.outline_mask = m1

    def handle_keydown(self, event):
        if event['key'] in self.key_down and self.key_down[event['key']] == True:
            return
        self.key_down[event['key']] = True

        ctrl = event['ctrlKey']

        if event['key'] == 'ArrowLeft':
            if ctrl:
                self.frame_slider.value = max(self.frame_min, self.frame - 10)
            else:
                self.frame_slider.value = max(self.frame_min, self.frame - 1)
        elif event['key'] == 'ArrowRight':
            if ctrl:
                self.frame_slider.value = min(self.frame_max, self.frame + 10)
            else:
                self.frame_slider.value = min(self.frame_max, self.frame + 1)
        elif event['key'] == 'c':
            channel = self.channel_slider.value + 1
            if channel > self.channel_max:
                channel = self.channel_min
            self.channel_slider.value = channel
        elif event['key'] == 'ArrowUp':
            index = self.particle_index()
            if index < len(self.all_particles) - 1:
                self.particle_dropdown.value = self.all_particles[index+1]
        elif event['key'] == 'ArrowDown':
            index = self.particle_index()
            if index > 0:
                self.particle_dropdown.value = self.all_particles[index-1]
        elif event['key'] == 'Enter' and ctrl:
            self.enabled_checkbox.value = not self.enabled_checkbox.value

    def handle_keyup(self, event):
        self.key_down[event['key']] = False

    def show(self):
        size = 400
        #self.image = widgets.Image(width = '60%', height = '60%', format='jpg')
        #self.image = widgets.Image(format='jpg', layout=widgets.Layout(width='100%', height='auto'))
        self.image = widgets.Image(format='jpg', layout=widgets.Layout(width='50%', height='auto', object_fit='contain'))

        # Plotting
        self.brightness_figure = bqplt.figure(title='Brightness')
        self.brightness_lines = bqplt.plot([],[])
        self.brightness_cursor_line = bqplt.vline(0, colors=[self.COLOR_RED])

        self.area_figure = bqplt.figure(title='Area')
        self.area_lines = bqplt.plot([],[])
        self.area_cursor_line = bqplt.vline(0, colors=[self.COLOR_RED])

        controls_widgets = []

        position_options = []
        for pos in self.positions:
            position_options.append((str(pos[0]), pos))

        self.position_dropdown = widgets.Dropdown(description='Position:')
        self.max_value_slider = widgets.IntSlider(min=0, max=np.iinfo(np.uint16).max, description='Max Pixel Value (Contrast)', value=self.max_pixel_value)
        #self.frame_slider = widgets.IntSlider(min=self.frame_min, max=self.frame_max, description='Frame', value=self.frame)
        self.frame_slider = widgets.IntSlider(description='Frame', value=self.frame)
        self.channel_slider = widgets.IntSlider(min=self.channel_min, max=self.channel_max, description='Channel', value=self.channel)
        self.particle_dropdown = widgets.Dropdown()
        self.enabled_checkbox = widgets.Checkbox(description='Cell Enabled')

        self.position_dropdown.observe(self.position_dropdown_changed, names='value')
        self.frame_slider.observe(self.frame_slider_changed, names='value')
        self.channel_slider.observe(self.channel_slider_changed, names='value')
        self.max_value_slider.observe(self.max_pixel_slider_changed, names='value')
        self.particle_dropdown.observe(self.particle_dropdown_changed, names='value')
        self.enabled_checkbox.observe(self.enabled_checkbox_changed, names='value')


        self.position_dropdown.options = position_options

        controls_widgets.append(self.position_dropdown)
        controls_widgets.append(self.max_value_slider)
        controls_widgets.append(self.frame_slider)
        controls_widgets.append(self.channel_slider)
        controls_widgets.append(self.particle_dropdown)
        controls_widgets.append(self.enabled_checkbox)

        controls_box = widgets.VBox(controls_widgets)

        self.area_figure.layout.height = '50%'
        self.brightness_figure.layout.height = '50%'
        plots_box = widgets.VBox([self.area_figure,self.brightness_figure], layout=widgets.Layout(width='50%'))


        #plots_box = widgets.VBox([self.area_figure,self.brightness_figure,controls_box], layout=widgets.Layout(width='50%', height='auto'))

        #output_box = widgets.VBox([plots_box,widgets.HBox([self.image,controls_box])])
        #output_box = widgets.VBox([widgets.HBox([self.image,plots_box]), controls_box])
        #output_box = widgets.HBox([self.image,plots_box])


        widgets.VBox([self.image,controls_box])
        output_box = widgets.VBox([widgets.HBox([self.image,plots_box]),controls_box])

        #output_box.layout.height = '90%'
        #output_box = widgets.HBox([widgets.VBox([self.image,controls_box], layout=widgets.Layout(width='50%')), plots_box])

        self.key_down = {}
        keyup_event = Event(source=output_box, watched_events=['keyup'], prevent_default_action=True)
        keyup_event.on_dom_event(self.handle_keyup)
        keydown_event = Event(source=output_box, watched_events=['keydown'], prevent_default_action=True)
        keydown_event.on_dom_event(self.handle_keydown)

        display(output_box)
