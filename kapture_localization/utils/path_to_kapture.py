# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import sys
import os.path as path
# when developing, prefer local kapture to the one installed on the system
HERE_PATH = path.normpath(path.dirname(__file__))
KATURE_REPO_PATH = path.normpath(path.join(HERE_PATH, '../../../kapture'))
# check the presence of kapture directory in repo to be sure its not the installed version
if path.isdir(path.join(KATURE_REPO_PATH, 'kapture')):
    # workaround for sibling import
    sys.path.insert(0, KATURE_REPO_PATH)
