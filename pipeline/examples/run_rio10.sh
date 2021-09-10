# Run this script in docker,
# but first pull the most recent version.

# docker pull kapture/kapture-localization
# docker run --runtime=nvidia -it --rm --volume <my_data>:<my_data> kapture/kapture-localization
# once the docker container is launched, go to your working directory of your choice (all data will be stored there)
# and run this script from there (of course you can also change WORKING_DIR=${PWD} to something else and run the script from somewhere else)

# 0a) Define paths
PYTHONBIN=python3.6
WORKING_DIR=${PWD}
DATASETS_PATH=${WORKING_DIR}/datasets
mkdir -p ${DATASETS_PATH}

TOPK=20  # number of retrieved images for mapping and localization
KPTS=20000 # number of local features to extract

# 0b) Download RIO10 dataset
# Note that you will be asked to accept or decline the license terms before download.
cd ${DATASETS_PATH}
kapture_download_dataset.py --install_path ${DATASETS_PATH} update
for SCENE in scene01 scene02 scene03 scene04 scene05 scene06 scene07 scene08 scene09 scene10; do 
  kapture_download_dataset.py --install_path ${DATASETS_PATH} install "RIO10_${SCENE}_*"
done

# 0c) Get extraction code for local and global features
# ! skip if already done !
# Deep Image retrieval - AP-GeM
pip3 install scikit-learn==0.22 torchvision==0.5.0 gdown tqdm
apt update
apt install unzip
cd ${WORKING_DIR}
git clone https://github.com/naver/deep-image-retrieval.git
cd deep-image-retrieval
mkdir -p dirtorch/data/
cd dirtorch/data/
gdown --id 1r76NLHtJsH-Ybfda4aLkUIoW3EEsi25I # downloads a pre-trained model of AP-GeM
unzip Resnet101-AP-GeM-LM18.pt.zip
rm -rf Resnet101-AP-GeM-LM18.pt.zip
# R2D2
cd ${WORKING_DIR}
git clone https://github.com/naver/r2d2.git

for SCENE in scene01 scene02 scene03 scene04 scene05 scene06 scene07 scene08 scene09 scene10; do 
  DATASET=RIO10/${SCENE}
  # 1) Create temporal mapping and query sets (they will be modified)
  mkdir -p ${WORKING_DIR}/${DATASET}/mapping/sensors
  cp ${DATASETS_PATH}/${DATASET}/mapping/sensors/*.txt ${WORKING_DIR}/${DATASET}/mapping/sensors/
  ln -s ${DATASETS_PATH}/${DATASET}/mapping/sensors/records_data ${WORKING_DIR}/${DATASET}/mapping/sensors/records_data

  mkdir -p ${WORKING_DIR}/${DATASET}/testing/sensors
  cp ${DATASETS_PATH}/${DATASET}/testing/sensors/*.txt ${WORKING_DIR}/${DATASET}/testing/sensors/
  ln -s ${DATASETS_PATH}/${DATASET}/testing/sensors/records_data ${WORKING_DIR}/${DATASET}/testing/sensors/records_data

  # 2) Merge mapping and testing kaptures (this will make it easier to extract the local and global features and it will be used for the localization step)
  kapture_merge.py -v debug -f \
    -i ${WORKING_DIR}/${DATASET}/mapping ${WORKING_DIR}/${DATASET}/testing \
    -o ${WORKING_DIR}/${DATASET}/map_plus_testing \
    --image_transfer link_relative

  # 3) Extract global features (we will use AP-GeM here)
  cd ${WORKING_DIR}/deep-image-retrieval
  ${PYTHONBIN} -m dirtorch.extract_kapture --kapture-root ${WORKING_DIR}/${DATASET}/map_plus_testing/ --checkpoint dirtorch/data/Resnet101-AP-GeM-LM18.pt --gpu 0
  # move to right location
  mkdir -p ${WORKING_DIR}/${DATASET}/global_features/Resnet101-AP-GeM-LM18/global_features
  mv ${WORKING_DIR}/${DATASET}/map_plus_testing/reconstruction/global_features/Resnet101-AP-GeM-LM18/* ${WORKING_DIR}/${DATASET}/global_features/Resnet101-AP-GeM-LM18/global_features/
  rm -rf ${WORKING_DIR}/${DATASET}/map_plus_testing/reconstruction/global_features/Resnet101-AP-GeM-LM18

  # 4) Extract local features (we will use R2D2 here)
  cd ${WORKING_DIR}/r2d2
  ${PYTHONBIN} extract_kapture.py --model models/r2d2_WASF_N8_big.pt --kapture-root ${WORKING_DIR}/${DATASET}/map_plus_testing/ --min-scale 0.3 --min-size 128 --max-size 9999 --top-k ${KPTS}
  # move to right location
  mkdir -p ${WORKING_DIR}/${DATASET}/local_features/r2d2_WASF_N8_big/descriptors
  mv ${WORKING_DIR}/${DATASET}/map_plus_testing/reconstruction/descriptors/r2d2_WASF_N8_big/* ${WORKING_DIR}/${DATASET}/local_features/r2d2_WASF_N8_big/descriptors/
  mkdir -p ${WORKING_DIR}/${DATASET}/local_features/r2d2_WASF_N8_big/keypoints
  mv ${WORKING_DIR}/${DATASET}/map_plus_testing/reconstruction/keypoints/r2d2_WASF_N8_big/* ${WORKING_DIR}/${DATASET}/local_features/r2d2_WASF_N8_big/keypoints/

  # 5) mapping pipeline
  LOCAL=r2d2_WASF_N8_big
  GLOBAL=Resnet101-AP-GeM-LM18
  kapture_pipeline_mapping.py -v debug -f \
    -i ${WORKING_DIR}/${DATASET}/mapping \
    -kpt ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/keypoints \
    -desc ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/descriptors \
    -gfeat ${WORKING_DIR}/${DATASET}/global_features/${GLOBAL}/global_features \
    -matches ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_no_gv/matches \
    -matches-gv ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_colmap_gv/matches \
    --colmap-map ${WORKING_DIR}/${DATASET}/colmap-sfm/${LOCAL}/${GLOBAL} \
    --topk ${TOPK}

  # 6) localization pipeline
  kapture_pipeline_localize.py -v debug -f \
    -i ${WORKING_DIR}/${DATASET}/mapping \
    --query ${WORKING_DIR}/${DATASET}/testing \
    -kpt ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/keypoints \
    -desc ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/descriptors \
    -gfeat ${WORKING_DIR}/${DATASET}/global_features/${GLOBAL}/global_features \
    -matches ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_no_gv/matches \
    -matches-gv ${WORKING_DIR}/${DATASET}/local_features/${LOCAL}/NN_colmap_gv/matches \
    --colmap-map ${WORKING_DIR}/${DATASET}/colmap-sfm/${LOCAL}/${GLOBAL} \
    -o ${WORKING_DIR}/${DATASET}/colmap-localize/${LOCAL}/${GLOBAL} \
    --topk ${TOPK} \
    --config 2 \
    --benchmark-style RIO10

  # 7) cat the output files in order to generate one file for benchmark submission
  cat ${WORKING_DIR}/${DATASET}/colmap-localize/${LOCAL}/${GLOBAL}/LTVL2020_style_result.txt >> ${WORKING_DIR}/RIO10/RIO10_LTVL2020_style_result_all_scenes_${LOCAL}_${GLOBAL}.txt
done
