# Run this script in docker,
# but first pull the most recent version.

# docker pull kapture/kapture-localization
# docker run --runtime=nvidia -it --rm --volume <my_data>:<my_data> kapture/kapture-localization
# once the docker container is launched, go to your working directory of your choice (all data will be stored there)
# and run this script from there (of course you can also change WORKING_DIR=${PWD} to something else and run the script from somewhere else)

###############################################
SCENES_DATASET_ROOT_URL="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"
################################################

# 0) Define paths and params
LOCAL_FEAT_DESC=r2d2_WASF_N8_big
LOCAL_FEAT_KPTS=20000 # number of local features to extract
GLOBAL_FEAT_DESC=Resnet101-AP-GeM-LM18
RETRIEVAL_TOPK=20  # number of retrieved images for mapping and localization

PYTHONBIN=python3
# select a working directory of your choice
WORKING_DIR=${PWD} 
TMP_DIR=${WORKING_DIR}/tmp/7seasons/
DATASETS_PATH=${WORKING_DIR}/datasets/7seasons
DATASET_NAMES=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")

# override vars for fast test
# uncomment the following to do a fastest test on subset with low quality parameters
# LOCAL_FEAT_DESC=faster2d2_WASF_N8_big
# LOCAL_FEAT_KPTS=5000 # number of local features to extract
# RETRIEVAL_TOPK=10  # number of retrieved images for mapping and localization
# DATASET_NAMES=("office")

LOCAL_FEAT_DIR=${LOCAL_FEAT_DESC}_${LOCAL_FEAT_KPTS}

# 0) install required tools
pip3 install scikit-learn==0.22 torchvision==0.5.0 gdown tqdm

# 1) Download, unzip, and convert dataset
mkdir -p ${DATASETS_PATH};
SCENES_DATASET_ZIP_URLS=()
for SCENE in ${DATASET_NAMES[*]}; do
  SCENES_DATASET_ZIP_URLS+=(${SCENES_DATASET_ROOT_URL}/${SCENE}.zip);
done

mkdir -p ${TMP_DIR};
cd ${TMP_DIR};
for ZIP_URL in ${SCENES_DATASET_ZIP_URLS[*]}; do
  wget ${ZIP_URL}
  ZIP_NAME=$(basename -- "${ZIP_URL}")
  if [ -f ${ZIP_NAME} ]; then
    unzip -o -q ${ZIP_NAME};
    rm ${ZIP_NAME};
  fi
done

for SCENE in ${DATASET_NAMES[*]}; do
  cd ${TMP_DIR}/${SCENE}
  unzip '*.zip'
done

# convert to kapture
for SCENE in ${DATASET_NAMES[*]}; do
  kapture_import_7scenes.py -v info \
    -i ${TMP_DIR}/${SCENE} \
    -o ${DATASETS_PATH}/${SCENE}/mapping \
    --image_transfer copy \
    -p 'mapping'
  kapture_import_7scenes.py -v info \
    -i ${TMP_DIR}/${SCENE} \
    -o ${DATASETS_PATH}/${SCENE}/query \
    --image_transfer copy \
    -p 'query'
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
              --min-scale 0.3 --min-size 128 --max-size 9999 --top-k ${LOCAL_FEAT_KPTS}
  ${PYTHONBIN} ${WORKING_DIR}/r2d2/extract_kapture.py --model ${WORKING_DIR}/r2d2/models/${LOCAL_FEAT_DESC}.pt \
              --kapture-root ${EXP_PATH}/query \
              --min-scale 0.3 --min-size 128 --max-size 9999 --top-k ${LOCAL_FEAT_KPTS}
done

# 4) mapping pipelines: RGBD and triangulation of local keypoint matches
for SCENE in ${DATASET_NAMES[*]}; do
  EXP_PATH=${DATASETS_PATH}/${SCENE}/${GLOBAL_FEAT_DESC}/${LOCAL_FEAT_DIR}

  # note: no image matches are needed when using RGBD data, the 3D points are directly taken from the depth map, thus, no need for triangulation 
  kapture_create_3D_model_from_depth.py -v debug -f \
    -i ${EXP_PATH}/mapping \
    -o ${EXP_PATH}/mapping_rgbd \
    -d kinect_depth_reg \
    --topk ${LOCAL_FEAT_KPTS} \
    --cellsizes "10" "5" "1" "0.1" "0.01" \
    --keypoints-type ${LOCAL_FEAT_DESC}
  
  rm -rf ${EXP_PATH}/mapping_rgbd/reconstruction/keypoints/*
  ln -s ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/keypoints ${EXP_PATH}/mapping_rgbd/reconstruction/keypoints/${LOCAL_FEAT_DESC}
  rm -rf ${EXP_PATH}/mapping_rgbd/reconstruction/descriptors/*
  ln -s ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/descriptors ${EXP_PATH}/mapping_rgbd/reconstruction/descriptors/${LOCAL_FEAT_DESC}

  kapture_export_colmap.py -v debug -f \
    -i ${EXP_PATH}/mapping_rgbd \
    -db ${EXP_PATH}/mapping_rgbd/colmap/colmap.db \
    -txt ${EXP_PATH}/mapping_rgbd/colmap/reconstruction \
    -kpt ${LOCAL_FEAT_DESC} \
    -desc ${LOCAL_FEAT_DESC}

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

  kapture_import_colmap.py -v debug -f \
    -db ${EXP_PATH}/mapping_triangulation/colmap/colmap.db \
    -txt ${EXP_PATH}/mapping_triangulation/colmap/reconstruction \
    -o ${EXP_PATH}/mapping_triangulation \
    -kpt ${LOCAL_FEAT_DESC} \
    -desc ${LOCAL_FEAT_DESC}
done

# 5) localization query
for SCENE in ${DATASET_NAMES[*]}; do
  EXP_PATH=${DATASETS_PATH}/${SCENE}/${GLOBAL_FEAT_DESC}/${LOCAL_FEAT_DIR}

  # create query-mapping pairs
  kapture_compute_image_pairs.py -v debug \
    --mapping ${EXP_PATH}/mapping \
    --query ${EXP_PATH}/query \
    --topk ${RETRIEVAL_TOPK} \
    -gfeat ${GLOBAL_FEAT_DESC} \
    -o ${EXP_PATH}/query_pairs_top${RETRIEVAL_TOPK}.txt

  # match query-mapping pairs
  kapture_merge.py -v debug \
    -i ${DATASETS_PATH}/${SCENE}/mapping ${DATASETS_PATH}/${SCENE}/query \
    -o ${EXP_PATH}/mapping_plus_query

  kapture_create_kapture_proxy.py -v debug -f \
    -i ${EXP_PATH}/mapping_plus_query \
    -o ${EXP_PATH}/mapping_plus_query_matches \
    -kpt ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/keypoints \
    -desc ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/descriptors \
    -gfeat ${DATASETS_PATH}/${SCENE}/global_features/${GLOBAL_FEAT_DESC}/global_features \
    -matches ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/NN_no_gv/matches \
    --keypoints-type ${LOCAL_FEAT_DESC} \
    --descriptors-type ${LOCAL_FEAT_DESC} \
    --global-features-type ${GLOBAL_FEAT_DESC}

  kapture_create_kapture_proxy.py -v debug -f \
    -i ${EXP_PATH}/mapping_plus_query \
    -o ${EXP_PATH}/mapping_plus_query_matches_gv \
    -kpt ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/keypoints \
    -desc ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/descriptors \
    -gfeat ${DATASETS_PATH}/${SCENE}/global_features/${GLOBAL_FEAT_DESC}/global_features \
    -matches ${DATASETS_PATH}/${SCENE}/local_features/${LOCAL_FEAT_DIR}/NN_colmap_gv/matches \
    --keypoints-type ${LOCAL_FEAT_DESC} \
    --descriptors-type ${LOCAL_FEAT_DESC} \
    --global-features-type ${GLOBAL_FEAT_DESC}

  kapture_compute_matches.py -v debug \
    -i ${EXP_PATH}/mapping_plus_query_matches \
    --pairsfile-path ${EXP_PATH}/query_pairs_top${RETRIEVAL_TOPK}.txt \
    -desc ${LOCAL_FEAT_DESC}

  kapture_run_colmap_gv.py -v debug -f \
    -i ${EXP_PATH}/mapping_plus_query_matches \
    -o ${EXP_PATH}/mapping_plus_query_matches_gv \
    --pairsfile-path ${EXP_PATH}/query_pairs_top${RETRIEVAL_TOPK}.txt \
    -kpt ${LOCAL_FEAT_DESC}

  # localize using pycolmap
  cp ${EXP_PATH}/mapping_rgbd/reconstruction/*.txt ${EXP_PATH}/mapping_plus_query_matches_gv/reconstruction/
  kapture_pycolmap_localize.py -v debug -f \
    -i ${EXP_PATH}/mapping_plus_query_matches_gv \
    --query ${DATASETS_PATH}/${SCENE}/query \
    -o ${EXP_PATH}/pycolmap-localize_rgbd \
    --pairsfile-path ${EXP_PATH}/query_pairs_top${RETRIEVAL_TOPK}.txt \
    --keypoints-type ${LOCAL_FEAT_DESC}
  rm ${EXP_PATH}/mapping_plus_query_matches_gv/reconstruction/*.txt
  cp ${EXP_PATH}/mapping_triangulation/reconstruction/*.txt ${EXP_PATH}/mapping_plus_query_matches_gv/reconstruction/
  kapture_pycolmap_localize.py -v debug -f \
    -i ${EXP_PATH}/mapping_plus_query_matches_gv \
    --query ${DATASETS_PATH}/${SCENE}/query \
    -o ${EXP_PATH}/pycolmap-localize_triangulation \
    --pairsfile-path ${EXP_PATH}/query_pairs_top${RETRIEVAL_TOPK}.txt \
    --keypoints-type ${LOCAL_FEAT_DESC} 

  # evaluate results
  kapture_evaluate.py -v debug -f \
    -i \
    ${EXP_PATH}/pycolmap-localize_rgbd \
    ${EXP_PATH}/pycolmap-localize_triangulation \
    --labels rgbd triangulation \
    -gt ${DATASETS_PATH}/${SCENE}/query \
    -o ${EXP_PATH}/eval
done