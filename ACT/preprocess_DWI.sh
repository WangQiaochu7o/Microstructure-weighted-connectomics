#!/bin/bash
set -e


########################################################################################
#
# Preprocessing script for FIREPATH DWI tractography data. 
# - Data should be in $SUBJECT_ID/DWI/raw folder, named 'ep2d_diff_3shells' 
# - raw folder should also contain .bvec and .bval files with the same name, and reverse
#       phase encoding data at b=0 with the same name plus _revPE.nii 
# - Change SUBJECT_DIR path if running on a different machine
########################################################################################




if [ $# -eq 0 ]; then echo '******** no subject ID provided'; exit 1; fi
subject=$1

SUBJECT_ID=$subject
SUBJECT_DIR=/home/localadmin/Documents/data/$SUBJECT_ID
DWI_DIR=$SUBJECT_DIR/DWI
RAW=ep2d_diff_3shells.nii

if [ $# -eq 2 ]; then NTHREADS=$2; else NTHREADS=24; fi
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$NTHREADS

if [ -d $SUBJECT_DIR ]; then echo ">>>>>>>>>>>> Subject" $SUBJECT_ID; else echo "Subject not found!!"; exit 1; fi

now=`date +'%Y%m%d_%H%M%S'`
LOG_FILE="DWI_preproc_${SUBJECT_ID}_${now}.log"
echo "Saving outputs to log file: [ $LOG_FILE ]"
exec > >(tee -i $DWI_DIR/$LOG_FILE)
exec 2>&1

echo ">>>>>>>>>>>>>>>>> Preprocessing DWI data for tractography"
echo "FSL HOME : $FSLDIR"
echo "FSL version" `flirt -version | awk '{ print $3 }'`
echo "MRtrix version" `mrview -version | head -n 1 | cut -d " " -f 3`

cd $DWI_DIR

###############################################################################
# MP-PCA
echo ">>>>>>>>>>>>>>>> MP-PCA Denoising "
mkdir denoise
dwidenoise -quiet -nthreads $NTHREADS -extent 5 -noise denoise/sigma.nii.gz raw/$RAW denoise/denoised.nii.gz

###############################################################################
# Gibbs unringing
echo ">>>>>>>>>>>>>>>> Gibbs Unringing "
mrdegibbs -quiet -nthreads $NTHREADS denoise/denoised.nii.gz denoise/denoised_degibbs.nii.gz

###############################################################################
# Topup
echo ">>>>>>>>>>>>>>>> TOPUP "
# make file with all b0 volumes (4xAP from 3 shell volume, 2xPA from revPE volume)
# verify b0 volumes are at indices 0 34 68 101
echo "Checking b0 indices"
IFS=' '
read -a bvs <<< `cat raw/${RAW::-4}.bval`
if [ $(( bvs[0] + ${bvs[34]} + ${bvs[68]} + ${bvs[101]} )) -ne 0 ]
then
	echo "ERROR! b0 VOLUMES NOT FOUND AT EXPECTED INDICES"
	exit 123
fi

echo "Compiling b0 volumes"
mkdir topup
for i in 0 34 68 101
do	
	fslroi denoise/denoised_degibbs.nii.gz topup/b0_normPE_$i $i 1
done
fslmerge -t topup/all_b0 topup/b0_normPE_0 topup/b0_normPE_34 topup/b0_normPE_68 topup/b0_normPE_101 raw/${RAW::-4}_revPE.nii
rm -f topup/b0_normPE_*

# Readout time doesn't matter as long as it is proportional
echo "Making acqparams.txt"
echo $'0 -1 0 0.05\n0 -1 0 0.05\n0 -1 0 0.05\n0 -1 0 0.05' > topup/acqparams.txt
echo $'0 1 0 0.05\n0 1 0 0.05' >> topup/acqparams.txt

echo "Running topup"
topup --imain=topup/all_b0 --datain=topup/acqparams.txt --config=b02b0.cnf --out=topup/topup

###############################################################################
# Eddy
echo ">>>>>>>>>>>>>>>> EDDY "

mkdir eddy
echo "Getting brain mask (note: just made from b0 volumes for processing, do not use as anatomical mask)"

fslroi topup/all_b0 eddy/b0_AP 0 4
fslmaths eddy/b0_AP -Tmean eddy/avg_b0_AP
mri_synthstrip -i eddy/avg_b0_AP.nii.gz -o eddy/avg_b0_brain.nii.gz -m eddy/avg_b0_brain_mask.nii.gz

echo "Making index.txt"
indx=""
for i in {1..102}; do indx="$indx 1"; done
echo $indx > eddy/index.txt

echo "Running eddy"
eddy --imain=denoise/denoised_degibbs --mask=eddy/avg_b0_brain_mask --acqp=topup/acqparams.txt --index=eddy/index.txt --bvecs=raw/${RAW::-4}.bvec --bvals=raw/${RAW::-4}.bval --topup=topup/topup --repol --out=eddy/denoised_degibbs_topup_eddy --cnr_maps

echo "Running eddyQC"
eddy_quad eddy/denoised_degibbs_topup_eddy -idx eddy/index.txt -par topup/acqparams.txt -m eddy/avg_b0_brain_mask -b raw/${RAW::-4}.bval

###############################################################################
# MSMT-CSD
echo ">>>>>>>>>>>>>>>> MSMT-CSD "
mkdir csd

echo "Estimating multishell response functions" 
dwi2response dhollander eddy/denoised_degibbs_topup_eddy.nii.gz csd/wm_response.txt csd/gm_response.txt csd/csf_response.txt -fslgrad raw/${RAW::-4}.bvec raw/${RAW::-4}.bval -mask eddy/avg_b0_brain_mask.nii.gz -quiet -nthreads $NTHREADS -shells 0,1000,2000,3000 -voxels csd/voxels.nii.gz

echo "Performing multi-shell mutli-tissue constrained spherical deconvolution"
dwi2fod msmt_csd eddy/denoised_degibbs_topup_eddy.nii.gz csd/wm_response.txt csd/wmfod.mif csd/gm_response.txt csd/gm.mif csd/csf_response.txt csd/csf.mif -fslgrad raw/${RAW::-4}.bvec raw/${RAW::-4}.bval -shells 0,1000,2000,3000 -nthreads $NTHREADS -mask eddy/avg_b0_brain_mask.nii.gz -quiet 

echo "Getting tissue fraction maps"
mrconvert -quiet -coord 3 0 csd/wmfod.mif - | mrcat -quiet csd/csf.mif csd/gm.mif - csd/vf.mif

mkdir tractography
echo "Compiling corrected b0 volumes"
for i in 0 34 68 101
do
        fslroi eddy/denoised_degibbs_topup_eddy.nii.gz tractography/b0_corr_$i $i 1
done
fslmerge -t tractography/all_b0 tractography/b0_corr_0 tractography/b0_corr_34 tractography/b0_corr_68 tractography/b0_corr_101
fslmaths tractography/all_b0 -Tmean tractography/dwi_regvol
rm -f tractography/b0_corr_* tractography/all_b0.nii.gz

mri_synthstrip -i tractography/dwi_regvol.nii.gz -o tractography/dwi_regvol_brain.nii.gz -m tractography/dwi_regvol_brain_mask.nii.gz

antsRegistration --dimensionality 3 --float 0 --output [../reg/dwi2anat,../reg/dwi2anat.nii.gz] --interpolation Linear --winsorize-image-intensities [0.01,0.99] --use-histogram-matching 0 --transform Rigid[0.0001] --metric MI[../T1/T1_brain.nii.gz,tractography/dwi_regvol_brain.nii.gz,1,32,Regular,0.25] --convergence [10000x10000x10000x10000,1e-6,10] --shrink-factors 4x3x2x1 --smoothing-sigmas 2x2x1x0vox

mtnormalise -mask tractography/dwi_regvol_brain_mask.nii.gz csd/wmfod.mif csd/wmfod_norm.mif csd/csf.mif csd/csf_norm.mif csd/gm.mif csd/gm_norm.mif

echo "** DWI PREPROCESSING DONE! **"
echo "  > Clean DWI data saved in:   eddy/denoised_degibbs_topup_eddy.nii.gz"
echo "  > WM FOD image:              csd/wmfod.mif"
echo "  > Run the following visual checks: "
echo "     * Check MP-PCA sigma map for anatomical structure (denoise/sigma.nii.gz)"
echo "     * Check clean DWI for susceptibility distortion correction and motion correction (eddy/denoised_degibbs_topup_eddy.nii.gz)"
echo "     * Check brain mask (eddy/avg_b0_brain_mask.nii.gz)"
echo "     * Check voxels used to estimate response functions (csd/voxels.nii.gz)"
echo "     * Check response functions (csd/{gm,wm,csd}_response.txt)"
echo "     * Check CSD tissue contribution map (csd/vf.mif)"
echo "     * General DWI quality control with eddyQC"
