from django import template

register = template.Library()

@register.inclusion_tag('components/analysis_controls.html')
def analysis_controls(n_channels, n_positions, n_frames):
    """Render analysis controls form"""
    return {
        'n_channels': n_channels,
        'n_positions': n_positions, 
        'n_frames': n_frames
    }

@register.inclusion_tag('components/analysis_scripts.html')
def analysis_scripts(n_channels, n_positions, n_frames):
    """Render analysis JavaScript"""
    return {
        'n_channels': n_channels,
        'n_positions': n_positions,
        'n_frames': n_frames
    }