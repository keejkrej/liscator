from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    # Page URLs
    path('', views.index, name='index'),
    path('test/', views.test, name='test'),
    path('select_paths/', views.select_paths, name='select_paths'),
    path('view/', views.view, name='view'),
    path('preprocess/', views.preprocess, name='preprocess'),
    path('documentation/', views.documentation, name='documentation'),
    path('analysis/', views.analysis, name='analysis'),
    
    # API URLs
    path('api/update_image/', views.update_image, name='update_image'),
    path('api/update_particle_enabled/', views.update_particle_enabled, name='update_particle_enabled'),
    path('api/do_segmentation/', views.do_segmentation, name='do_segmentation'),
    path('api/do_tracking/', views.do_tracking, name='do_tracking'),
    path('api/do_square_rois/', views.do_square_rois, name='do_square_rois'),
    path('api/do_export/', views.do_export, name='do_export'),
    path('api/list_directory/', views.list_directory, name='list_directory'),
    path('api/select_folder/', views.select_folder, name='select_folder'),
]