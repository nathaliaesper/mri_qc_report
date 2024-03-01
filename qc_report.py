import numpy as np
import os
import sys
import glob
from tkinter import Tcl

import pandas as pd
import nilearn
from nilearn import plotting
from nilearn.maskers import NiftiSpheresMasker, NiftiMasker
from nilearn.plotting import plot_roi
from nilearn import image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import PyPDF2
from PyPDF2 import PdfReader, PdfWriter, PdfMerger


def create_seed_masker(coord, functional_img_path):

	seed_masker = NiftiSpheresMasker(
		coord,
		radius=8,
		detrend=True,
		standardize="zscore_sample",
		standardize_confounds="zscore_sample",
		low_pass=0.1,
		high_pass=0.01,
		t_r=2,
		memory="nilearn_cache",
		memory_level=1,
		verbose=0,
	)
	
	seed_timeseries = seed_masker.fit_transform(functional_img_path)
	
	return seed_timeseries


def create_functional_masker(functional_img_path):
	
	functional_masker = NiftiMasker(
		smoothing_fwhm = 6,
		detrend = True,
		standardize = "zscore_sample",
		standardize_confounds = "zscore_sample",
		low_pass = 0.1,
		high_pass = 0.01,
		t_r = 2,
		memory = "nilearn_cache",
		memory_level = 1,
		verbose = 0,
	)
	
	functional_timeseries = functional_masker.fit_transform(functional_img_path)
	
	return functional_masker, functional_timeseries


def calculate_correlation(brain_timeseries, seed_timeseries):
	corr = (np.dot(brain_timeseries.T, seed_timeseries) / seed_timeseries.shape[0])
	
	return corr


def display_network_axial(seed_to_voxel_correlations, seed_coord, functional_mask, subj_func_name, axes_idx, axes, coord_idx):
	filename = subj_func_name.split("/func/")
	filename = filename[1]
	seed_to_voxel_correlations_img = functional_mask.inverse_transform(seed_to_voxel_correlations.T)

	display = plotting.plot_stat_map(
		seed_to_voxel_correlations_img,
		threshold = 0.3,
		vmax = 1,
		display_mode = "z",
		cut_coords = 7,
		axes = axes[axes_idx]
	)
	display.title(os.path.basename(subj_func_name), size = 10)
	display.add_markers(marker_coords = seed_coord, marker_color = "g", marker_size = 50)


def display_network_sagittal(seed_to_voxel_correlations, seed_coord, functional_mask, subj_func_name, axes_idx, axes, coord_idx):
	seed_to_voxel_correlations_img = functional_mask.inverse_transform(seed_to_voxel_correlations.T)

	display = plotting.plot_stat_map(
		seed_to_voxel_correlations_img,
		threshold = 0.3,
		vmax = 1,
		display_mode = "x",
		cut_coords = 7,
		axes = axes[axes_idx]
	)
	display.add_markers(marker_coords = seed_coord, marker_color = "g", marker_size = 50)


def read_xcp_file(filename, fmri_preproc_step):
	
	if fmri_preproc_step == 'norm':
		xcp_filepath = filename.split('preproc_bold.nii.gz')[0]+'xcp_quality.tsv'
		xcp_file = pd.read_csv(xcp_filepath, sep = '\t')
	else:
		xcp_filepath = filename.split('space-T1w_sbref.nii')[0]+'space-MNI152NLin6ASym_reg-aCompCor_desc-xcp_quality.tsv'
		xcp_file = pd.read_csv(xcp_filepath, sep = '\t')
	
	values = {'Dice': None, 'Jaccard': None, 'CrossCorr': None, 'Coverage': None}
	values['Dice'] = str(round(xcp_file[fmri_preproc_step+'Dice'].tolist()[0],4))
	values['Jaccard'] = str(round(xcp_file[fmri_preproc_step+'Jaccard'].tolist()[0],4))
	values['CrossCorr'] = str(round(xcp_file[fmri_preproc_step+'CrossCorr'].tolist()[0],4))
	values['Coverage'] = str(round(xcp_file[fmri_preproc_step+'Coverage'].tolist()[0],4))
   
	return values


def print_xcp_values(xcp_values, ax):
	text_to_display = str(xcp_values).replace('{', '').replace('}', '').replace("'" , "  ").replace(", ", " | ")
	axes[ax].annotate(text_to_display,
			xy=(0.5, 0.3),
			horizontalalignment='center', verticalalignment='bottom')

	axes[ax].axis('off')


def create_structural_report(subject_path):
	subj = 'sub-NDARJJ173BRX'
	logo_image = "ChildMindInstitute_Logo_Horizontal_OneColor_Blue_RGB.png"
	func_norm_pattern = subj + '/ses-*/func/*task-rest*reg-*desc-preproc_bold.nii.gz'
	func_coreg_pattern = subj + '/ses-*/func/*task-rest*space-T1w_sbref.nii.gz'
	anat_norm_pattern = subj + '/ses-*/anat/*desc-head_T1w.nii.gz'
	anat_coreg_pattern = subj + '/ses-*/anat/*desc-head_T1w.nii.gz'

	func_norm = glob.glob(func_norm_pattern)
	func_norm_count = (np.size(func_norm) * 2) + np.size(func_norm)

	func_coreg = glob.glob(func_coreg_pattern)
	func_coreg_count = (np.size(func_coreg) * 2) + np.size(func_coreg)

	fig, axes = plt.subplots(func_norm_count+func_coreg_count+1, 1, figsize=(20,30))

	img = mpimg.imread(logo_image)
	axes[0].imshow(img, aspect = 'equal')
	axes[0].axis('off')
	axes[0].set_xlim(left=1)

	axes_idx = 1

	for anat_file in glob.glob(anat_norm_pattern):
		if anat_file.find('space') != -1:
			for func_norm in Tcl().call('lsort', '-dict', func_norm):
				xcp_values = read_xcp_file(func_norm, 'norm')
				text_to_display = 'Spatial Normalization - ' + str(xcp_values).replace('{', '').replace('}', '').replace("'" , "  ").replace(", ", " | ")
				axes[axes_idx].annotate(text_to_display,xy=(0.5, 0.3),horizontalalignment='center', verticalalignment='bottom')
				axes[axes_idx].axis('off')
				
				func_norm_first = image.index_img(func_norm, 0)
				axes_idx += 1
				d = plot_roi(func_norm_first, anat_file, display_mode = 'z', cut_coords = [-24, -10, 0, 10, 24, 46, 55], alpha = 0.3, threshold = 0, axes = axes[axes_idx])
				d.title(os.path.basename(func_norm), size = 10)
				plot_roi(func_norm_first, anat_file, display_mode = 'x', cut_coords = [50, 38, 15, 0, -15, -38, -50], alpha = 0.3, threshold = 0, axes = axes[axes_idx+1])
				axes_idx += 2

	for anat_cor_file in glob.glob(anat_coreg_pattern):
		if anat_cor_file.find('space') == -1:
			for func_coreg in Tcl().call('lsort', '-dict', func_coreg):
				xcp_values = read_xcp_file(func_coreg, 'coreg')
				text_to_display = 'Corregistration - ' + str(xcp_values).replace('{', '').replace('}', '').replace("'" , "  ").replace(", ", " | ")
				axes[axes_idx].annotate(text_to_display, xy=(0.5, 0.3),horizontalalignment='center', verticalalignment='bottom')
				axes[axes_idx].axis('off')

				axes_idx += 1
				
				d = plot_roi(func_coreg, anat_cor_file, display_mode = 'z', cut_coords = [-46, -24, -10, 0, 10, 24, 46], alpha = 0.2, threshold = 0, axes = axes[axes_idx])
				d.title(os.path.basename(func_coreg), size = 10)
				plot_roi(func_coreg, anat_cor_file, display_mode = 'x', cut_coords = [50, 38, 15, 0, -15, -38, -50], alpha = 0.2, threshold = 0, axes = axes[axes_idx+1])
				axes_idx += 2

				
	plt.subplots_adjust(wspace=0, hspace=0)

	split_file = func_coreg.split("_")
	filename = split_file[0] + '_' + split_file[1] + '_desc-masks_quality.pdf'

	fig.savefig(filename, dpi=1200)


def create_functional_report(subject_path):
	subj = 'sub-NDARJJ173BRX'
	logo_image = "ChildMindInstitute_Logo_Horizontal_OneColor_Blue_RGB.png"

	func_pattern = subj + '/ses-*/func/*task-rest*space-*desc-preproc_bold.nii.gz'
	functional_data = glob.glob(func_pattern)

	pcc_coords = [(0, -52, 18)]
	auditory_coords = [(-54, -14, 8)]
	coords = [pcc_coords, auditory_coords]

	functional_count = (np.size(functional_data) * len(coords)) * 2

	fig, axes = plt.subplots(functional_count+1, 1, figsize=(20,22))
	img = mpimg.imread(logo_image)
	axes[0].imshow(img, aspect = 'equal')
	axes[0].axis('off')
	axes[0].set_xlim(left=1)

	axes_idx = 1


	for subj_func_name in functional_data:
		for coord in coords:
			seed_timeseries = create_seed_masker(coord, subj_func_name)
			functional_mask, functional_timeseries = create_functional_masker(subj_func_name)
			seed_to_voxel_correlations = calculate_correlation(functional_timeseries, seed_timeseries)
			display_network_axial(seed_to_voxel_correlations, coord, functional_mask, subj_func_name, axes_idx, axes, coord)
			display_network_sagittal(seed_to_voxel_correlations, coord, functional_mask, subj_func_name, axes_idx+1, axes, coord)
			axes_idx += 2

	# Remove subplot borders
	plt.subplots_adjust(wspace=0, hspace=0)

	split_file = functional_data[0].split("_")
	filename = split_file[0] + '_' + split_file[1] + '_desc-functionalnetworks_quality.pdf'

	fig.savefig(filename, dpi=1200)


def create_segmentation_report(subject_path):
	subj = 'sub-NDARJJ173BRX'
	logo_image = "ChildMindInstitute_Logo_Horizontal_OneColor_Blue_RGB.png"

	anatomical_pattern = subj + '/ses-*/anat/*desc-preproc_T1w.nii.gz'
	segmentation_pattern = subj + '/ses-*/anat/*_label*_mask.nii.gz'

	segmentation_files = glob.glob(segmentation_pattern)
	segmentation_count = np.size(segmentation_files) * 2

	fig, axes = plt.subplots(7, 1, figsize=(20,20))

	img = mpimg.imread(logo_image)
	axes[0].imshow(img, aspect = 'equal')
	axes[0].axis('off')
	axes[0].set_xlim(left=1)

	axes_idx = 1

	for anat_file in glob.glob(anatomical_pattern):
		if anat_file.find('space') == -1 :
			for segmentation_image in Tcl().call('lsort', '-dict', segmentation_files):
				if segmentation_image.find('desc') == -1 | segmentation_image.find('probseg') == -1:
					d = plot_roi(segmentation_image, bg_img=anat_file, display_mode='z', cmap='GnBu', alpha=1.0, axes = axes[axes_idx])
					d.title(os.path.basename(segmentation_image), size = 10)

					plot_roi(segmentation_image, bg_img = anat_file, display_mode = 'x', cmap = 'GnBu', alpha = 1.0, axes = axes[axes_idx+1])
					axes_idx += 2

				
	plt.subplots_adjust(wspace=0, hspace=0)

	split_file = segmentation_files[0].split("_")
	filename = split_file[0] + '_' + split_file[1] + '_desc-segmentation_quality.pdf'

	fig.savefig(filename, dpi=1200)


subject_path = sys.argv[1]
print("Creating structural report")
create_structural_report(subject_path)
print("Creating functional report")
create_functional_report(subject_path)
print("Creating segmentation report")
create_segmentation_report(subject_path)
