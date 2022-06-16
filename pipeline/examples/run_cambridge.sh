# Run this script in docker,
# but first pull the most recent version.

# docker pull kapture/kapture-localization
# docker run --runtime=nvidia -it --rm --volume <my_data>:<my_data> kapture/kapture-localization
# once the docker container is launched, go to your working directory of your choice (all data will be stored there)
# and run this script from there (of course you can also change WORKING_DIR=${PWD} to something else and run the script from somewhere else)

###############################################
declare -A DOWNLOAD_LINKS=(\
  ["KingsCollege"]="https://www.repository.cam.ac.uk/bitstream/handle/1810/251342/KingsCollege.zip?sequence=4&isAllowed=y" \
  ["OldHospital"]="https://www.repository.cam.ac.uk/bitstream/handle/1810/251340/OldHospital.zip?sequence=4&isAllowed=y" \
  ["StMarysChurch"]="https://www.repository.cam.ac.uk/bitstream/handle/1810/251294/StMarysChurch.zip?sequence=5&isAllowed=y" \
  ["GreatCourt"]="https://www.repository.cam.ac.uk/bitstream/handle/1810/251291/GreatCourt.zip?sequence=4&isAllowed=y" \
  ["ShopFacade"]="https://www.repository.cam.ac.uk/bitstream/handle/1810/251336/ShopFacade.zip?sequence=4&isAllowed=y" \
  ["Street"]="https://www.repository.cam.ac.uk/bitstream/handle/1810/251292/Street.zip?sequence=5&isAllowed=y" )
# 0) Define paths and params
LOCAL_FEAT_DESC=r2d2_WASF_N8_big
LOCAL_FEAT_KPTS=20000 # number of local features to extract
LOCAL_FEAT_MAX_SIZE=9999 # maximum image size to be considered for local feature extraction
GLOBAL_FEAT_DESC=Resnet101-AP-GeM-LM18
RETRIEVAL_TOPK=20  # number of retrieved images for mapping and localization

PYTHONBIN=python3
# select a working directory of your choice
WORKING_DIR=${PWD} 
TMP_DIR=${WORKING_DIR}/tmp/cambridge/
DATASETS_PATH=${WORKING_DIR}/datasets/cambridge
DATASET_NAMES=("ShopFacade" "KingsCollege" "OldHospital" "StMarysChurch" "GreatCourt"  "Street")

# override vars for fast test
# uncomment the following to do a fastest test on subset with low quality parameters
# LOCAL_FEAT_DESC=faster2d2_WASF_N8_big
# LOCAL_FEAT_KPTS=1000 # number of local features to extract
# LOCAL_FEAT_MAX_SIZE=1000 # maximum image size to be considered for local feature extraction
# RETRIEVAL_TOPK=5  # number of retrieved images for mapping and localization
# DATASET_NAMES=("StMarysChurch")

LOCAL_FEAT_DIR=${LOCAL_FEAT_DESC}_${LOCAL_FEAT_KPTS}

# 0) install required tools
pip3 install scikit-learn==0.22 torchvision==0.5.0 gdown tqdm

# 1) Download, unzip, and convert dataset
mkdir -p ${DATASETS_PATH};
mkdir -p ${TMP_DIR};
cd ${TMP_DIR};
for LANDMARK in "${!DOWNLOAD_LINKS[@]}"; do
  wget -O ${LANDMARK}.zip ${DOWNLOAD_LINKS[${LANDMARK}]}
  if [ -f ${LANDMARK}.zip ]; then
    unzip -o -q ${LANDMARK}.zip;
    rm ${SCENE}.zip;
  fi
done

# convert to kapture
for SCENE in ${DATASET_NAMES[*]}; do
  sed 's/.jpg/.png/g' ${TMP_DIR}/${SCENE}/reconstruction.nvm > ${TMP_DIR}/${SCENE}/reconstruction_png.nvm
  tail -n +4 ${TMP_DIR}/${SCENE}/dataset_train.txt > ${TMP_DIR}/${SCENE}/dataset_train_cut.txt
  cut -d\  -f1 ${TMP_DIR}/${SCENE}/dataset_train_cut.txt > ${TMP_DIR}/${SCENE}/dataset_train_list.txt
  tail -n +4 ${TMP_DIR}/${SCENE}/dataset_test.txt > ${TMP_DIR}/${SCENE}/dataset_test_cut.txt
  cut -d\  -f1 ${TMP_DIR}/${SCENE}/dataset_test_cut.txt > ${TMP_DIR}/${SCENE}/dataset_test_list.txt

  kapture_import_nvm.py -v info -f \
    -i ${TMP_DIR}/${SCENE}/reconstruction_png.nvm \
    -im ${TMP_DIR}/${SCENE}/ \
    -o ${DATASETS_PATH}/${SCENE}/mapping \
    --filter-list ${TMP_DIR}/${SCENE}/dataset_train_list.txt
  kapture_import_nvm.py -v info -f \
    -i ${TMP_DIR}/${SCENE}/reconstruction_png.nvm \
    -im ${TMP_DIR}/${SCENE}/ \
    -o ${DATASETS_PATH}/${SCENE}/query \
    --filter-list ${TMP_DIR}/${SCENE}/dataset_test_list.txt
done

# create proxy kapture versions of mapping and query that will be linked to the local and global features
# see https://github.com/naver/kapture-localization/blob/main/doc/tutorial.adoc#recommended-dataset-structure
for SCENE in ${DATASET_NAMES[*]}; do
  EXP_PATH=${DATASETS_PATH}/${SCENE}/${GLOBAL_FEAT_DESC}/${LOCAL_FEAT_DIR}
  mkdir -p ${DATASETS_PATH}/${SCENE}/global_features/${GLOBAL_FEAT_DESC}/global_features
  mkdir -p ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/descriptors
  mkdir -p ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/keypoints
  mkdir -p ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/NN_no_gv/matches
  mkdir -p ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/NN_colmap_gv/matches
  kapture_create_kapture_proxy.py -v debug -f \
    -i ${DATASETS_PATH}/${SCENE}/mapping \
    -o ${EXP_PATH}/mapping \
    -kpt ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/keypoints \
    -desc ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/descriptors \
    -gfeat ${DATASETS_PATH}/${SCENE}/global_features/${GLOBAL_FEAT_DESC}/global_features \
    --keypoints-type ${LOCAL_FEAT_DESC} \
    --descriptors-type ${LOCAL_FEAT_DESC} \
    --global-features-type ${GLOBAL_FEAT_DESC}
  
  kapture_create_kapture_proxy.py -v debug -f \
    -i ${DATASETS_PATH}/${SCENE}/query \
    -o ${EXP_PATH}/query \
    -kpt ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/keypoints \
    -desc ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/descriptors \
    -gfeat ${DATASETS_PATH}/${SCENE}/global_features/${GLOBAL_FEAT_DESC}/global_features \
    --keypoints-type ${LOCAL_FEAT_DESC} \
    --descriptors-type ${LOCAL_FEAT_DESC} \
    --global-features-type ${GLOBAL_FEAT_DESC}
done

# Note that we will now delete the tmp data, including the downloaded archives!
rm -rf ${TMP_DIR}

# 2) Extract global features (we will use AP-GeM here)
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
for SCENE in ${DATASET_NAMES[*]}; do
  EXP_PATH=${DATASETS_PATH}/${SCENE}/${GLOBAL_FEAT_DESC}/${LOCAL_FEAT_DIR}

  ${PYTHONBIN} -m dirtorch.extract_kapture --kapture-root ${EXP_PATH}/mapping \
  --checkpoint ${WORKING_DIR}/deep-image-retrieval/dirtorch/data/${GLOBAL_FEAT_DESC}.pt --gpu 0
  ${PYTHONBIN} -m dirtorch.extract_kapture --kapture-root ${EXP_PATH}/query \
  --checkpoint ${WORKING_DIR}/deep-image-retrieval/dirtorch/data/${GLOBAL_FEAT_DESC}.pt --gpu 0
done

# 3) Extract local features (we will use R2D2 here)
cd ${WORKING_DIR}
git clone https://github.com/naver/r2d2.git
for SCENE in ${DATASET_NAMES[*]}; do
  EXP_PATH=${DATASETS_PATH}/${SCENE}/${GLOBAL_FEAT_DESC}/${LOCAL_FEAT_DIR}

  ${PYTHONBIN} ${WORKING_DIR}/r2d2/extract_kapture.py --model ${WORKING_DIR}/r2d2/models/${LOCAL_FEAT_DESC}.pt \
              --kapture-root ${EXP_PATH}/mapping \
              --min-scale 0.3 --min-size 128 --max-size ${LOCAL_FEAT_MAX_SIZE} --top-k ${LOCAL_FEAT_KPTS}
  ${PYTHONBIN} ${WORKING_DIR}/r2d2/extract_kapture.py --model ${WORKING_DIR}/r2d2/models/${LOCAL_FEAT_DESC}.pt \
              --kapture-root ${EXP_PATH}/query \
              --min-scale 0.3 --min-size 128 --max-size ${LOCAL_FEAT_MAX_SIZE} --top-k ${LOCAL_FEAT_KPTS}
done

# 4) mapping pipeline
for SCENE in ${DATASET_NAMES[*]}; do
  EXP_PATH=${DATASETS_PATH}/${SCENE}/${GLOBAL_FEAT_DESC}/${LOCAL_FEAT_DIR}

  kapture_pipeline_mapping.py -v debug -f \
    -i ${DATASETS_PATH}/${SCENE}/mapping \
    -kpt ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/keypoints \
    -desc ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/descriptors \
    -gfeat ${DATASETS_PATH}/${SCENE}/global_features/${GLOBAL_FEAT_DESC}/global_features \
    -matches ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/NN_no_gv/matches \
    -matches-gv ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/NN_colmap_gv/matches \
    --colmap-map ${EXP_PATH}/mapping_triangulation/colmap \
    --topk ${RETRIEVAL_TOPK} \
    --keypoints-type ${LOCAL_FEAT_DESC} \
    --descriptors-type ${LOCAL_FEAT_DESC} \
    --global-features-type ${GLOBAL_FEAT_DESC}
done

# 5) localization query
for SCENE in ${DATASET_NAMES[*]}; do
  EXP_PATH=${DATASETS_PATH}/${SCENE}/${GLOBAL_FEAT_DESC}/${LOCAL_FEAT_DIR}

  kapture_pipeline_localize.py -v debug -f \
  -i ${DATASETS_PATH}/${SCENE}/mapping \
  --query ${DATASETS_PATH}/${SCENE}/query \
  -kpt ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/keypoints \
  -desc ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/descriptors \
  -gfeat ${DATASETS_PATH}/${SCENE}/global_features/${GLOBAL_FEAT_DESC}/global_features \
  -matches ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/NN_no_gv/matches \
  -matches-gv ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/NN_colmap_gv/matches \
  --colmap-map ${EXP_PATH}/mapping_triangulation/colmap \
  -o ${EXP_PATH}/colmap-localize \
  --topk ${RETRIEVAL_TOPK} \
  --config 2 \
  --keypoints-type ${LOCAL_FEAT_DESC} \
  --descriptors-type ${LOCAL_FEAT_DESC} \
  --global-features-type ${GLOBAL_FEAT_DESC} \
  -s export_LTVL2020
done