#!/bin/bash
set -e


########################################################################################
#
#
#
########################################################################################


if [ $# -eq 0 ]; then echo '******** no subject ID provided'; exit 1; fi
subject=$1

SUBJECT_ID=$subject
SUBJECT_DIR=/home/localadmin/Documents/data/$SUBJECT_ID

if [ -n $2 ]; then NTHREADS=$2; else NTHREADS=24; fi
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$NTHREADS

if [ -d $SUBJECT_DIR ]; then echo ">>>>>>>>>>>> Subject" $SUBJECT_ID; else echo "Subject not found!!"; exit 1; fi
cd $SUBJECT_DIR/DWI

now=`date +'%Y%m%d_%H%M%S'`
LOG_FILE="tractography_${SUBJECT_ID}_${now}.log"
echo "Saving outputs to log file: [ $LOG_FILE ]"
exec > >(tee -i $SUBJECT_DIR/DWI/$LOG_FILE)
exec 2>&1

echo "Subject $SUBJECT_ID : $SUBJECT_DIR"
echo "MRtrix version" `mrview -version | head -n 1 | cut -d " " -f 3`

cd $SUBJECT_DIR/DWI

echo "Processing structural data for tractography"

antsApplyTransforms -d 3 -e 0 -i ../T1/fs_labelled_dk_fixed.nii.gz -r tractography/dwi_regvol_brain.nii.gz -o tractography/fs_dwi_fslabels.nii.gz -n NearestNeighbor -t [../reg/dwi2anat0GenericAffine.mat,1]
labelconvert tractography/fs_dwi_fslabels.nii.gz $FREESURFER_HOME/FreeSurferColorLUT.txt ~/mrtrix3/share/mrtrix3/labelconvert/fs_default.txt tractography/fs_dwi_mrtrixlabels.nii.gz -quiet -nthreads $NTHREADS
5ttgen freesurfer -lut $FREESURFER_HOME/FreeSurferColorLUT.txt tractography/fs_dwi_fslabels.nii.gz tractography/5tt_dwi.nii.gz -nocrop -quiet -nthreads $NTHREADS
5tt2gmwmi tractography/5tt_dwi.nii.gz tractography/gmwmi.nii.gz -nthreads $NTHREADS

echo ">>>>>>>>>>>>>>>>>>> Running tractography"
tckgen -act tractography/5tt_dwi.nii.gz -backtrack -seed_gmwmi tractography/gmwmi.nii.gz -select 10000000 -angle 45 -minlength 10 -maxlength 200 -cutoff 0.05 -step 1 csd/wmfod_norm.mif tractography/tracks10M.tck -quiet -nthreads $NTHREADS
echo ">>>>>>>>>>>>>>>>>>> Filtering"
tcksift -act tractography/5tt_dwi.nii.gz -nthreads $NTHREADS -quiet -term_number 1000000 tractography/tracks10M.tck csd/wmfod_norm.mif tractography/tracks_sift_1M.tck


