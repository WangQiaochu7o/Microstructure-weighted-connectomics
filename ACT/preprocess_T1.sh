#!/bin/bash
set -e


########################################################################################
#
# Preprocessing script for FIREPATH T1 data. 
#  - Data should be organised into $SUBJECT_ID/T1/raw/$RAW where RAW ends in .nii,
#  - Run ./preprocess_T1.sh $SUBJECT_ID $RAW $RUNFREESURFER ($NTHREADS optional)
#      e.g. ./preprocess_T1.sh Aim1_sub-04 t1_mp2rage_sag_p3_iso_UNI.nii 1 20
#
########################################################################################

if [ $# -eq 0 ]; then echo '******** no subject ID provided'; exit 1; fi
SUBJECT_ID=$1
SUBJECT_DIR=/home/localadmin/Documents/data/$SUBJECT_ID
RAW=$2
RUNFS=$3
if [ $# -eq $4 ]; then NTHREADS=$4; else NTHREADS=24; fi
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$NTHREADS

if [ -d $SUBJECT_DIR ]; then echo ">>>>>>>>>>>> Subject" $SUBJECT_ID; else echo "Subject not found!!"; exit 1; fi
if [ -f $SUBJECT_DIR/T1/raw/$RAW ]; then echo ">>>>>>>>>>>> Raw data" $RAW; else echo "Raw data not found in data folder!!"; exit 1; fi

echo ">>>>>>>>>>>>>>>>> Processing T1"

now=`date +'%Y%m%d_%H%M%S'`
LOG_FILE="T1_preproc_${SUBJECT_ID}_${now}.log"
echo "Saving outputs to log file: [ $LOG_FILE ]"
exec > >(tee -i $SUBJECT_DIR/T1/$LOG_FILE)
exec 2>&1

echo "FSL HOME : $FSLDIR"
echo "FSL version" `flirt -version | awk '{ print $3 }'`
if [ $RUNFS -eq 1 ]
then
	echo "Running FreeSurfer version" `recon-all -version | awk '{ print $1 }'`
else
	echo "Not running FreeSurfer (flag = $RUNFS). Set to 1 to run."
fi

###############################################################################

cd $SUBJECT_DIR/T1

echo ">>>>>>>>>>>>>>>>> Denoising"
DenoiseImage -d 3 -i raw/$RAW -o [T1_denoised.nii.gz,sigma.nii.gz] -v

echo ">>>>>>>>>>>>>>>>> Skull stripping"
mri_synthstrip -i T1_denoised.nii.gz -o T1_brain.nii.gz -m T1_brain_mask.nii.gz -g  --no-csf

echo ">>>>>>>>>>>>>>>>> Registration"
mkdir ../reg
antsRegistrationSyN.sh -d 3 -f /home/localadmin/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz -m T1_brain.nii.gz -o ../reg/anat2std -n $NTHREADS

echo ">>>>>>>>>>>>>>>>> Tissue segmentation"
fast T1_brain
fslmaths T1_brain_pve_2 -thr 1 -bin T1_wmseg
fslmaths T1_brain_seg -uthr 1.1 -bin T1_csfseg
fslmaths T1_brain_seg -thr 1.9 -uthr 2.1 -bin T1_gmseg

if [ $RUNFS -eq 1 ]
then
	echo ">>>>>>>>>>>>>>>>> FreeSurfer"
	recon-all -subjid fs-parc -sd $SUBJECT_DIR/T1 -all -threads $NTHREADS # -i T1_denoised.nii.gz -all -threads $NTHREADS

	mri_label2vol --seg fs-parc/mri/aparc+aseg.mgz --temp fs-parc/mri/rawavg.mgz --o fs_labelled_dk.mgz --regheader fs-parc/mri/aparc+aseg.mgz
	labelsgmfix fs_labelled_dk.mgz T1_brain.nii.gz $FREESURFER_HOME/FreeSurferColorLUT.txt fs_labelled_dk_fixed.nii.gz -premasked -sgm_amyg_hipp -nthreads $NTHREADS -quiet
	labelconvert fs_labelled_dk_fixed.nii.gz $FREESURFER_HOME/FreeSurferColorLUT.txt ~/mrtrix3/share/mrtrix3/labelconvert/fs_default.txt fs_labelled_dk_fixed_mrtrix.nii.gz -quiet

	echo ">>>>>>>>>>>>>>>>> T1 preproc done."

else
	echo ">>>>>>>>>>>>>>>>> T1 preproc done (FreeSurfer not run)."
fi