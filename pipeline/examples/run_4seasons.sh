# Run this script in docker,
# but first pull the most recent version.

# docker pull kapture/kapture-localization
# docker run --runtime=nvidia -it --rm --volume <my_data>:<my_data> kapture/kapture-localization
# once the docker container is launched, go to your working directory of your choice (all data will be stored there)
# and run this script from there (of course you can also change WORKING_DIR=${PWD} to something else and run the script from somewhere else)


# GOTO https://vision.cs.tum.edu/webshare/g/4seasons-dataset/html/form.php
# fill the form, agree to license, ad you should be able
# then go to https://vision.cs.tum.edu/data/datasets/4seasons-dataset/download
# fill password and get the links

###############################################
SEASONS_DATASET_ROOT_URL=
SEASONS_DATASET_RELOCATION_ROOT_URL="https://github.com/pmwenzel/mlad-iccv2021/raw/main/"
################################################

if [ -z "$SEASONS_DATASET_ROOT_URL" ]; then
  echo "Please fill in the root url for dataset (SEASONS_DATASET_ROOT_URL)."
  echo "You may ask permission to dataset owner for that."
  exit 0
fi

# 0) Define paths and params
LOCAL_FEAT_DESC=r2d2_WASF_N8_big
LOCAL_FEAT_KPTS=20000 # number of local features to extract
GLOBAL_FEAT_DESC=Resnet101-AP-GeM-LM18
GLOBAL_FEAT_TOPK=20  # number of retrieved images for mapping and localization

PYTHONBIN=python3
WORKING_DIR=${PWD}
TMP_DIR=/tmp/4seasons/
DATASETS_PATH=${WORKING_DIR}/datasets/4seasons
DATASET_NAMES=("countryside" "neighborhood" "old_town")
DATASET_MAPPING=("recording_2020-10-08_09-57-28" "recording_2021-02-25_13-25-15" "recording_2020-10-08_11-53-41")
DATASET_VALIDATION=("recording_2020-06-12_11-26-43" "recording_2020-12-22_11-54-24" "recording_2021-01-07_10-49-45")
DATASET_QUERY=("recording_2021-01-07_14-03-57" "recording_2021-05-10_18-26-26" "recording_2021-05-10_19-51-14")
DATASET_RELOCATION=("relocalizationFile_recording_2020-10-08_09-57-28_to_recording_2021-01-07_14-03-57.txt" \
               "relocalizationFile_recording_2021-02-25_13-25-15_to_recording_2021-05-10_18-26-26.txt" \
               "relocalizationFile_recording_2020-10-08_11-53-41_to_recording_2021-05-10_19-51-14.txt" )
#DATASET_ALL=("${DATASET_MAPPING[@]}" "${DATASET_QUERY[@]}")


# override vars for fast test
# uncomment the following to do a fastest test on subset with low quality parameters.
#LOCAL_FEAT_DESC=faster2d2_WASF_N8_big
#LOCAL_FEAT_KPTS=200 # number of local features to extract
#GLOBAL_FEAT_TOPK=5  # number of retrieved images for mapping and localization
#DATASET_NAMES=("countryside")
#DATASET_MAPPING=("recording_2020-10-08_09-57-28")
#DATASET_VALIDATION=("recording_2020-06-12_11-26-43")
#DATASET_QUERY=("recording_2021-01-07_14-03-57")
#DATASET_RELOCATION=("relocalizationFile_recording_2020-10-08_09-57-28_to_recording_2021-01-07_14-03-57.txt")
DATASET_ALL=("${DATASET_MAPPING[@]}" "${DATASET_VALIDATION[@]}" "${DATASET_QUERY[@]}")

# 0) install required tools
pip3 install scikit-learn==0.22 torchvision==0.5.0 gdown tqdm


# 1) Download dataset
if [ ! -d ${DATASETS_PATH} ]; then
  mkdir -p ${DATASETS_PATH};
  SEASONS_DATASET_ZIP_URLS=()
  SEASONS_DATASET_ZIP_URLS+=(${SEASONS_DATASET_ROOT_URL}/calibration/calibration.zip)
  for RECORD in ${DATASET_ALL[*]}; do
    # download poses
    SEASONS_DATASET_ZIP_URLS+=(${SEASONS_DATASET_ROOT_URL}/dataset/${RECORD}/${RECORD}_reference_poses.zip);
    # download images
    SEASONS_DATASET_ZIP_URLS+=(${SEASONS_DATASET_ROOT_URL}/dataset/${RECORD}/${RECORD}_stereo_images_undistorted.zip);
  done

  # download from official and unzip it
  mkdir -p ${TMP_DIR};
  cd ${TMP_DIR};
  for ZIP_URL in ${SEASONS_DATASET_ZIP_URLS[*]}; do
    wget ${ZIP_URL}
    ZIP_NAME=$(basename -- "${ZIP_URL}")
    if [ -f ${ZIP_NAME} ]; then
      unzip -o -q ${ZIP_NAME};
      rm ${ZIP_NAME};
    fi
  done
  # convert to kapture
  for RECORD in ${DATASET_ALL[*]}; do
    kapture_import_4seasons.py -v info -i ${TMP_DIR}/${RECORD} \
                                       -o ${DATASETS_PATH}/records/${RECORD} \
                                       --image_transfer copy
  done
  rm -rf ${TMP_DIR}
fi

# 2) reorganize by place and mapping/query split
if [ ! -d ${DATASETS_PATH}/places ]; then
  mkdir -p ${DATASETS_PATH}/places
  cd ${DATASETS_PATH}/places

  # do not copy, but symlink sensors/
  # DO NOT DELETE records/
  for i in "${!DATASET_NAMES[@]}"; do
    mkdir -p ${DATASETS_PATH}/places/${DATASET_NAMES[i]}/{mapping,query};
    ln -s ../../../records/${DATASET_MAPPING[i]}/sensors ${DATASET_NAMES[i]}/mapping/sensors;
    ln -s ../../../records/${DATASET_VALIDATION[i]}/sensors ${DATASET_NAMES[i]}/validation/sensors;
    ln -s ../../../records/${DATASET_QUERY[i]}/sensors ${DATASET_NAMES[i]}/query/sensors;

    kapture_merge.py -v info \
    -i ${DATASETS_PATH}/places/${DATASET_NAMES[i]}/mapping \
       ${DATASETS_PATH}/places/${DATASET_NAMES[i]}/validation \
       ${DATASETS_PATH}/places/${DATASET_NAMES[i]}/query \
    -o ${DATASETS_PATH}/places/${DATASET_NAMES[i]}/mapping_validation_query \
    --image_transfer link_relative
  done
fi

# 3) Extract global features (we will use AP-GeM here)
# Deep Image retrieval - AP-GeM
if [ ! -d ${WORKING_DIR}/deep-image-retrieval ]; then
  cd ${WORKING_DIR}
  git clone https://github.com/naver/deep-image-retrieval.git
fi

# downloads a pre-trained model of AP-GeM
if [ ! -f ${WORKING_DIR}/deep-image-retrieval/dirtorch/data/${GLOBAL_FEAT_DESC}.pt ]; then
  mkdir -p ${WORKING_DIR}/deep-image-retrieval/dirtorch/data/
  cd ${WORKING_DIR}/deep-image-retrieval/dirtorch/data/
  gdown --id 1r76NLHtJsH-Ybfda4aLkUIoW3EEsi25I
  unzip ${GLOBAL_FEAT_DESC}.pt.zip
  rm -f ${GLOBAL_FEAT_DESC}.pt.zip
fi

cd ${WORKING_DIR}/deep-image-retrieval
for PLACE in ${DATASET_NAMES[*]}; do
  ${PYTHONBIN} -m dirtorch.extract_kapture --kapture-root ${DATASETS_PATH}/places/${PLACE}/mapping_validation_query/ \
  --checkpoint ${WORKING_DIR}/deep-image-retrieval/dirtorch/data/${GLOBAL_FEAT_DESC}.pt --gpu 0

  # move global features to right location
  # see https://github.com/naver/kapture-localization/blob/main/doc/tutorial.adoc#recommended-dataset-structure
  mkdir -p ${DATASETS_PATH}/places/${PLACE}/global_features/${GLOBAL_FEAT_DESC}/global_features
  mv ${DATASETS_PATH}/places/${PLACE}/mapping_validation_query/reconstruction/global_features/${GLOBAL_FEAT_DESC}/* \
     ${DATASETS_PATH}/places/${PLACE}/global_features/${GLOBAL_FEAT_DESC}/global_features
  rm -rf ${WORKING_DIR}/${DATASET}/mapping_validation_query/reconstruction/global_features/${GLOBAL_FEAT_DESC}
done

##################################################################
# 4) Extract local features (we will use R2D2 here)
cd ${WORKING_DIR}
git clone https://github.com/naver/r2d2.git
for PLACE in ${DATASET_NAMES[*]}; do
  ${PYTHONBIN} ${WORKING_DIR}/r2d2/extract_kapture.py --model ${WORKING_DIR}/r2d2/models/${LOCAL_FEAT_DESC}.pt \
              --kapture-root ${DATASETS_PATH}/places/${PLACE}/mapping_validation_query/ \
              --min-scale 0.3 --min-size 128 --max-size 1000 --top-k ${LOCAL_FEAT_KPTS}  #< change max-size 9999

  # move keypoints and descriptors to right location
  # see https://github.com/naver/kapture-localization/blob/main/doc/tutorial.adoc#recommended-dataset-structure
  mkdir -p ${DATASETS_PATH}/places/${PLACE}/local_features/${LOCAL_FEAT_DESC}/descriptors
  mv ${DATASETS_PATH}/places/${PLACE}/mapping_validation_query/reconstruction/descriptors/${LOCAL_FEAT_DESC}/* \
     ${DATASETS_PATH}/places/${PLACE}/local_features/${LOCAL_FEAT_DESC}/descriptors
  mkdir -p ${DATASETS_PATH}/places/${PLACE}/local_features/${LOCAL_FEAT_DESC}/keypoints
  mv ${DATASETS_PATH}/places/${PLACE}/mapping_validation_query/reconstruction/keypoints/${LOCAL_FEAT_DESC}/* \
     ${DATASETS_PATH}/places/${PLACE}/local_features/${LOCAL_FEAT_DESC}/keypoints/
  rm -rf ${DATASETS_PATH}/places/${PLACE}/mapping_validation_query/reconstruction/descriptors/${LOCAL_FEAT_DESC}
  rm -rf ${DATASETS_PATH}/places/${PLACE}/mapping_validation_query/reconstruction/keypoints/${LOCAL_FEAT_DESC}
done

# 6) mapping pipeline
for PLACE in ${DATASET_NAMES[*]}; do
  kapture_pipeline_mapping.py -v debug -f \
    -i ${DATASETS_PATH}/places/${PLACE}/mapping \
    -kpt ${DATASETS_PATH}/places/${PLACE}/local_features/${LOCAL_FEAT_DESC}/keypoints \
    -desc ${DATASETS_PATH}/places/${PLACE}/local_features/${LOCAL_FEAT_DESC}/descriptors \
    -gfeat ${DATASETS_PATH}/places/${PLACE}/global_features/${GLOBAL_FEAT_DESC}/global_features \
    -matches ${DATASETS_PATH}/places/${PLACE}/local_features/${LOCAL_FEAT_DESC}/NN_no_gv/matches \
    -matches-gv ${DATASETS_PATH}/places/${PLACE}/local_features/${LOCAL_FEAT_DESC}/NN_colmap_gv/matches \
    --colmap-map ${DATASETS_PATH}/places/${PLACE}/colmap-sfm/${LOCAL_FEAT_DESC}/${GLOBAL_FEAT_DESC} \
    --topk ${GLOBAL_FEAT_TOPK}
done

# 7) localization validation
for PLACE in ${DATASET_NAMES[*]}; do
  kapture_pipeline_localize.py -v debug -f \
    -i ${DATASETS_PATH}/places/${PLACE}/mapping \
    --query ${DATASETS_PATH}/places/${PLACE}/validation \
    -kpt ${DATASETS_PATH}/places/${PLACE}/local_features/${LOCAL_FEAT_DESC}/keypoints \
    -desc ${DATASETS_PATH}/places/${PLACE}/local_features/${LOCAL_FEAT_DESC}/descriptors \
    -gfeat ${DATASETS_PATH}/places/${PLACE}/global_features/${GLOBAL_FEAT_DESC}/global_features \
    -matches ${DATASETS_PATH}/places/${PLACE}/local_features/${LOCAL_FEAT_DESC}/NN_no_gv/matches \
    -matches-gv ${DATASETS_PATH}/places/${PLACE}/local_features/${LOCAL_FEAT_DESC}/NN_colmap_gv/matches \
    --colmap-map ${DATASETS_PATH}/places/${PLACE}/colmap-sfm/${LOCAL_FEAT_DESC}/${GLOBAL_FEAT_DESC} \
    -o ${DATASETS_PATH}/places/${PLACE}/colmap-localize/${LOCAL_FEAT_DESC}/${GLOBAL_FEAT_DESC} \
    --topk ${GLOBAL_FEAT_TOPK} \
    --config 2

  # you may find results in:
  cat ${DATASETS_PATH}/places/${PLACE}/colmap-localize/${LOCAL_FEAT_DESC}/${GLOBAL_FEAT_DESC}/eval/stats.txt
done

# 8) localization query
for i in "${!DATASET_NAMES[@]}"; do
  PLACE=${DATASET_NAMES[i]}
  kapture_pipeline_localize.py -v debug -f \
    -i ${DATASETS_PATH}/places/${PLACE}/mapping \
    --query ${DATASETS_PATH}/places/${PLACE}/query \
    -kpt ${DATASETS_PATH}/places/${PLACE}/local_features/${LOCAL_FEAT_DESC}/keypoints \
    -desc ${DATASETS_PATH}/places/${PLACE}/local_features/${LOCAL_FEAT_DESC}/descriptors \
    -gfeat ${DATASETS_PATH}/places/${PLACE}/global_features/${GLOBAL_FEAT_DESC}/global_features \
    -matches ${DATASETS_PATH}/places/${PLACE}/local_features/${LOCAL_FEAT_DESC}/NN_no_gv/matches \
    -matches-gv ${DATASETS_PATH}/places/${PLACE}/local_features/${LOCAL_FEAT_DESC}/NN_colmap_gv/matches \
    --colmap-map ${DATASETS_PATH}/places/${PLACE}/colmap-sfm/${LOCAL_FEAT_DESC}/${GLOBAL_FEAT_DESC} \
    -o ${DATASETS_PATH}/places/${PLACE}/colmap-localize/${LOCAL_FEAT_DESC}/${GLOBAL_FEAT_DESC} \
    --topk ${GLOBAL_FEAT_TOPK} \
    --config 2

  # download the pair files and format result for 4seasons benchmark
  RELOCATION_FILE_URL=${SEASONS_DATASET_RELOCATION_ROOT_URL}/${DATASET_RELOCATION[i]}
  wget ${SEASONS_DATASET_RELOCATION_ROOT_URL}/${DATASET_RELOCATION[i]} \
      -O ${DATASETS_PATH}/places/${PLACES}/mapping/${DATASET_RELOCATION[i]}
  kapture_export_4seasons_pairs.py -v debug -f \
    -i ${DATASETS_PATH}/places/${PLACE}/mapping
    -p ${DATASETS_PATH}/places/${PLACES}/mapping/${DATASET_RELOCATION[i]}
    -o ${DATASETS_PATH}/places/${PLACES}/${DATASET_RELOCATION[i]}

  # you may find relocation to send to dataset benchmark in:
  echo ${DATASETS_PATH}/places/${PLACES}/${DATASET_RELOCATION[i]}
done