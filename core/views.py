from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
import json
import os
import sys

# Add old directory to path to import Flask utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'old'))

try:
    from gui import CellViewer
    import pyama_util
except ImportError:
    CellViewer = None
    pyama_util = None

# HashMap to store CellViewer instances by user session
user_cell_viewers = {}

def cleanup_expired_viewers():
    """Remove CellViewer instances for expired sessions"""
    from django.contrib.sessions.models import Session
    
    # Get all active session keys
    active_sessions = set(Session.objects.filter(
        expire_date__gt=timezone.now()
    ).values_list('session_key', flat=True))
    
    # Remove viewers for expired sessions
    expired_keys = set(user_cell_viewers.keys()) - active_sessions
    for key in expired_keys:
        if key in user_cell_viewers:
            # Cleanup heavy resources
            viewer = user_cell_viewers[key]
            if hasattr(viewer, 'cleanup'):
                viewer.cleanup()
            del user_cell_viewers[key]
    
    return len(expired_keys)

def get_user_id(request):
    """Get unique ID for anonymous user using session key"""
    if not request.session.session_key:
        request.session.create()  # Create session if doesn't exist
    return request.session.session_key

def get_cell_viewer(request):
    """Get CellViewer instance for current user"""
    user_id = get_user_id(request)
    return user_cell_viewers.get(user_id)

def create_cell_viewer(request, nd2_path, output_path, init_type):
    """Create a new CellViewer instance for current user"""
    if not CellViewer:
        return None
    
    user_id = get_user_id(request)
    user_cell_viewers[user_id] = CellViewer(
        nd2_path=nd2_path,
        output_path=output_path,
        init_type=init_type
    )
    return user_cell_viewers[user_id]

def index(request):
    return render(request, 'pages/index.html')

def test(request):
    return JsonResponse({'message': 'test'})

@csrf_exempt
def select_paths(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        nd2_path = data['nd2_path']
        out_path = data['out_path']
        redirect_to = data['redirect_to']

        if not nd2_path or not out_path:
            return JsonResponse({'error': 'Both ND2 path and output path must be selected'}, status=400)

        if CellViewer:
            init_type = 'view' if redirect_to == 'view' else 'analysis'
            create_cell_viewer(request, nd2_path, out_path, init_type)

        if redirect_to == 'view':
            return JsonResponse({'redirect': '/view/'})
        elif redirect_to == 'analysis':
            return JsonResponse({'redirect': '/analysis/'})
        else:
            return JsonResponse({'redirect': '/'})

def view(request):
    cell_viewer = get_cell_viewer(request)
    if cell_viewer is None:
        return redirect('core:index')
    
    cell_viewer.position_changed()
    current_particle_index = cell_viewer.particle_index()
    
    context = {
        'channel_image': cell_viewer.return_image(),
        'n_positions': len(cell_viewer.positions),
        'n_channels': cell_viewer.channel_max,
        'n_frames': cell_viewer.frame_max,
        'all_particles_len': cell_viewer.all_particles_len,
        'current_particle_index': current_particle_index,
        'brightness_plot': cell_viewer.brightness_plot,
        'disabled_particles': cell_viewer.disabled_particles
    }
    return render(request, 'pages/view.html', context)

def preprocess(request):
    return render(request, 'pages/preprocess.html')

def documentation(request):
    svg = "static/images/UserTutorial.svg"
    return render(request, 'pages/documentation.html', {'svg': svg})

@csrf_exempt
def update_image(request):
    if request.method == 'POST':
        cell_viewer = get_cell_viewer(request)
        if not cell_viewer:
            return JsonResponse({'error': 'Cell viewer not initialized'}, status=400)
        new_position = int(request.POST['position'])
        new_channel = int(request.POST['channel'])
        new_frame = int(request.POST['frame'])
        new_particle = int(request.POST['particle'])

        if new_position != cell_viewer.position:
            cell_viewer.position = cell_viewer.position_options[new_position]
            cell_viewer.position_changed()

        if new_particle != cell_viewer.particle:
            cell_viewer.particle = cell_viewer.all_particles[new_particle]
            cell_viewer.particle_changed()

        cell_viewer.channel = new_channel
        cell_viewer.frame = new_frame

        cell_viewer.get_channel_image()
        cell_viewer.draw_outlines()
        
        return JsonResponse({
            'channel_image': cell_viewer.return_image(),
            'brightness_plot': cell_viewer.brightness_plot,
            'all_particles_len': cell_viewer.all_particles_len,
            'particle_enabled': cell_viewer.particle_enabled,
            'current_particle': cell_viewer.particle,
            'disabled_particles': cell_viewer.disabled_particles
        })

@csrf_exempt
def update_particle_enabled(request):
    if request.method == 'POST':
        cell_viewer = get_cell_viewer(request)
        if not cell_viewer:
            return JsonResponse({'error': 'Cell viewer not initialized'}, status=400)
        data = json.loads(request.body)
        enabled = data['enabled']
        cell_viewer.particle_enabled = enabled
        cell_viewer.particle_enabled_changed()
        
        return JsonResponse({
            'channel_image': cell_viewer.return_image(),
            'brightness_plot': cell_viewer.brightness_plot,
            'all_particles_len': cell_viewer.all_particles_len,
            'disabled_particles': cell_viewer.disabled_particles
        })
    return JsonResponse({'error': 'Cell viewer not initialized'}, status=400)

@csrf_exempt
def do_segmentation(request):
    if request.method == 'POST':
        cell_viewer = get_cell_viewer(request)
        if not cell_viewer or not pyama_util:
            return JsonResponse({'error': 'Cell viewer not initialized'}, status=400)
        data = json.loads(request.body)
        nd2_path = cell_viewer.nd2_path
        out_dir = cell_viewer.output_path
        positions = list(range(data['position_min'], data['position_max'] + 1))
        frame_min = data['frame_min']
        frame_max = data['frame_max']

        segmentation_channel = []
        fluorescence_channels = []
        for i in range(cell_viewer.channel_max + 1):
            channel_type = data[f'channel_{i}']
            if channel_type == 'Brightfield':
                segmentation_channel.append(i)
            elif channel_type == 'Fluorescent':
                fluorescence_channels.append(i)

        segmentation_channel = segmentation_channel[0] if len(segmentation_channel) == 1 else segmentation_channel

        pyama_util.segment_positions(nd2_path, out_dir, positions, segmentation_channel, fluorescence_channels, frame_min=frame_min, frame_max=frame_max)
        return JsonResponse({'status': 'success'})

@csrf_exempt
def do_tracking(request):
    if request.method == 'POST':
        cell_viewer = get_cell_viewer(request)
        if not cell_viewer or not pyama_util:
            return JsonResponse({'error': 'Cell viewer not initialized'}, status=400)
        data = json.loads(request.body)
        out_dir = cell_viewer.output_path
        positions = list(range(data['position_min'], data['position_max'] + 1))
        expand_labels = data['expand_labels']

        pyama_util.tracking_pyama(out_dir, positions, expand=expand_labels)
        return JsonResponse({'status': 'success'})

@csrf_exempt
def do_square_rois(request):
    if request.method == 'POST':
        cell_viewer = get_cell_viewer(request)
        if not cell_viewer or not pyama_util:
            return JsonResponse({'error': 'Cell viewer not initialized'}, status=400)
        data = json.loads(request.body)
        out_dir = cell_viewer.output_path
        positions = list(range(data['position_min'], data['position_max'] + 1))
        square_um_size = data['square_size']

        pyama_util.square_roi(out_dir, positions, square_um_size)
        return JsonResponse({'status': 'success'})

@csrf_exempt
def do_export(request):
    if request.method == 'POST':
        cell_viewer = get_cell_viewer(request)
        if not cell_viewer or not pyama_util:
            return JsonResponse({'error': 'Cell viewer not initialized'}, status=400)
        data = json.loads(request.body)
        out_dir = cell_viewer.output_path
        positions = list(range(data['position_min'], data['position_max'] + 1))
        minutes = data['minutes']

        try:
            pyama_util.csv_output(out_dir, positions, minutes)
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

def analysis(request):
    cell_viewer = get_cell_viewer(request)
    if cell_viewer is None:
        return redirect('core:index')
    
    n_positions = len(cell_viewer.nd2.metadata['fields_of_view'])+1
    context = {
        'n_positions': n_positions,
        'n_channels': cell_viewer.channel_max,
        'n_frames': cell_viewer.frame_max
    }
    return render(request, 'pages/analysis.html', context)

def list_directory(request):
    path = request.GET.get('path', '/')
    try:
        items = os.listdir(path)
        return JsonResponse({
            'path': path,
            'items': [{'name': item, 'isDirectory': os.path.isdir(os.path.join(path, item))} for item in items]
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
def select_folder(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        path = data['path']
        return JsonResponse({'message': f'Folder selected: {path}'})