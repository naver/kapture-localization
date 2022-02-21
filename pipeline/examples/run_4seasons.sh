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
SEASONS_DATASET_ROOT_URL="https://XXXXXXXXXXXXXXXXXXXX"
################################################

# 0a) Define paths and params
TOPK=20  # number of retrieved images for mapping and localization
LOCAL_FEAT=r2d2_WASF_N8_big
#LOCAL_FEAT_KPTS=20000 # number of local features to extract
LOCAL_FEAT_KPTS=2000 # number of local features to extract
GLOBAL_FEAT=Resnet101-AP-GeM-LM18

PYTHONBIN=python3.6
WORKING_DIR=${PWD}
TMP_DIR=/tmp/4seasons/
DATASETS_PATH=${WORKING_DIR}/datasets/4seasons
DATASET_NAMES=("countryside" "neighborhood" "old_town")
DATASET_MAPPING=("recording_2020-10-08_09-57-28" "recording_2021-02-25_13-25-15" "recording_2020-10-08_11-53-41")
DATASET_QUERY=("recording_2021-01-07_14-03-57" "recording_2021-05-10_18-26-26" "recording_2021-05-10_19-51-14")
DATASET_ALL=("${DATASET_MAPPING[@]}" "${DATASET_QUERY[@]}")

#rm -rf ${DATASETS_PATH}/${DATASET}
# 1) Download dataset
# Note that you will be asked to accept or decline the license terms before download.
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

# create aliases
if [ ! -d ${DATASETS_PATH}/places ]; then
  mkdir -p ${DATASETS_PATH}/places
  cd ${DATASETS_PATH}/places

  for i in "${!DATASET_NAMES[@]}"; do
    for PART in "mapping" "query"; do
      mkdir -p ${DATASETS_PATH}/places/${DATASET_NAMES[i]}/$PART;
      ln -s ../../../records/${DATASET_MAPPING[i]}/sensors ${DATASET_NAMES[i]}/$PART/sensors;
    done;

    kapture_merge.py -v info \
    -i ${DATASETS_PATH}/places/${DATASET_NAMES[i]}/mapping \
       ${DATASETS_PATH}/places/${DATASET_NAMES[i]}/query \
    -o ${DATASETS_PATH}/places/${DATASET_NAMES[i]}/mapping_plus_query \
    --image_transfer link_relative

    # prepare a place to host local features common for mapping, query and mapping_plus_query
    mkdir -p ${DATASETS_PATH}/places/${DATASET_NAMES[i]}/global_features/${GLOBAL_FEAT}/global_features;
    for PART in "mapping" "query" "mapping_plus_query"; do
      mkdir -p ${DATASETS_PATH}/places/${DATASET_NAMES[i]}/${PART}/reconstruction;
      ln -s ../../global_features \
            ${DATASETS_PATH}/places/${DATASET_NAMES[i]}/${PART}/reconstruction/global_features;
    done;
  done
fi

# 2) Create temporal mapping and query sets (they will be modified)
# 4) Extract global features (we will use AP-GeM here)
# Deep Image retrieval - AP-GeM
cd ${WORKING_DIR}
if [ ! -d ${WORKING_DIR}/deep-image-retrieval ]; then
  pip3 install scikit-learn==0.22 torchvision==0.5.0 gdown tqdm
  cd ${WORKING_DIR}
  git clone https://github.com/naver/deep-image-retrieval.git
  cd deep-image-retrieval
  mkdir -p dirtorch/data/
  cd dirtorch/data/
  if [ ! -f ${GLOBAL_FEAT}.pt ]; then
    gdown --id 1r76NLHtJsH-Ybfda4aLkUIoW3EEsi25I # downloads a pre-trained model of AP-GeM
    unzip ${GLOBAL_FEAT}.pt.zip
    rm -f ${GLOBAL_FEAT}.pt.zip
  fi
fi

cd ${WORKING_DIR}/deep-image-retrieval
for PLACE in ${DATASET_NAMES[*]}; do
  ${PYTHONBIN} -m dirtorch.extract_kapture --kapture-root ${DATASETS_PATH}/places/${PLACE}/mapping_plus_query/ --checkpoint dirtorch/data/${GLOBAL_FEAT}.pt --gpu 0
done

exit 0
##################################################################
##################################################################
##################################################################
# move to right location
mkdir -p ${WORKING_DIR}/${DATASET}/global_features/${GLOBAL}/global_features
mv ${WORKING_DIR}/${DATASET}/map_plus_query/reconstruction/global_features/${GLOBAL}/* ${WORKING_DIR}/${DATASET}/global_features/${GLOBAL}/global_features/
rm -rf ${WORKING_DIR}/${DATASET}/map_plus_query/reconstruction/global_features/${GLOBAL}

# 5) Extract local features (we will use R2D2 here)
cd ${WORKING_DIR}
git clone https://github.com/naver/r2d2.git
cd ${WORKING_DIR}/r2d2
${PYTHONBIN} extract_kapture.py --model models/${LOCAL}.pt --kapture-root ${WORKING_DIR}/${DATASET}/map_plus_query/ --min-scale 0.3 --min-size 128 --max-size 9999 --top-k ${KPTS}
# move to right location
mkdir -p ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/descriptors
mv ${WORKING_DIR}/${DATASET}/map_plus_query/reconstruction/descriptors/${LOCAL}/* ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/descriptors/
mkdir -p ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/keypoints
mv ${WORKING_DIR}/${DATASET}/map_plus_query/reconstruction/keypoints/${LOCAL}/* ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/keypoints/

# 6) mapping pipeline
kapture_pipeline_mapping.py -v debug -f \
  -i ${WORKING_DIR}/${DATASET}/mapping \
  -kpt ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/keypoints \
  -desc ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/descriptors \
  -gfeat ${WORKING_DIR}/${DATASET}/global_features/${GLOBAL}/global_features \
  -matches ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_no_gv/matches \
  -matches-gv ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_colmap_gv/matches \
  --colmap-map ${WORKING_DIR}/${DATASET}/colmap-sfm/${LOCAL}/${GLOBAL} \
  --topk ${TOPK}

# 7) localization pipeline
kapture_pipeline_localize.py -v debug -f \
  -i ${WORKING_DIR}/${DATASET}/mapping \
  --query ${WORKING_DIR}/${DATASET}/query \
  -kpt ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/keypoints \
  -desc ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/descriptors \
  -gfeat ${WORKING_DIR}/${DATASET}/global_features/${GLOBAL}/global_features \
  -matches ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_no_gv/matches \
  -matches-gv ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_colmap_gv/matches \
  --colmap-map ${WORKING_DIR}/${DATASET}/colmap-sfm/${LOCAL}/${GLOBAL} \
  -o ${WORKING_DIR}/${DATASET}/colmap-localize/${LOCAL}/${GLOBAL} \
  --topk ${TOPK} \
  --config 2
