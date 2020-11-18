# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import sys
import os.path as path

# when developing, prefer local kapture to the one installed on the system
HERE_PATH = path.abspath(path.normpath(path.dirname(__file__)))

KATURE_LOCALIZATION_REPO_PATH = path.normpath(path.join(HERE_PATH, '../'))
# check the presence of kapture directory in repo to be sure its not the installed version
if path.isdir(path.join(KATURE_LOCALIZATION_REPO_PATH, 'kapture_localization')):
    # workaround for sibling import
    sys.path.insert(0, KATURE_LOCALIZATION_REPO_PATH)


# KATURE_LOCALIZATION_TOOLS_PATH = path.normpath(path.join(HERE_PATH, '../'))
# # check the presence of pipeline directory in repo to be sure its not the installed version
# if path.isdir(path.join(KATURE_LOCALIZATION_TOOLS_PATH, 'pipeline')):
#     # workaround for sibling import
#     sys.path.insert(0, KATURE_LOCALIZATION_TOOLS_PATH)
