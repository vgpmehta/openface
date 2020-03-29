import os
import shutil

try:
    from urllib.request import urlopen # Python 3.x
    from urllib.error import URLError, HTTPError
except ImportError:
    from urllib2 import urlopen # Python 2
    from urllib2 import URLError, HTTPError


# Core OpenFace model files to download from GitHub
    
repo_base = 'https://raw.githubusercontent.com/TadasBaltrusaitis/OpenFace/master/'
base_modpath = 'lib/local/LandmarkDetector/'
classifier_path = 'lib/3rdParty/OpenCV/classifiers/'
classifier_file = 'haarcascade_frontalface_alt.xml'

model_files = {
    'detection_validation': {
        'validator_cnn.txt': None,
        'validator_cnn_68.txt': None,
        'validator_general_68.txt': None
    },
    'mtcnn_detector': {
        'MTCNN_detector.txt': None,
        'ONet.dat': None,
        'PNet.dat': None,
        'RNet.dat': None
    },
    'pdms': {
        'In-the-wild_aligned_PDM_68.txt': None,
        'Multi-PIE_aligned_PDM_68.txt': None,
        'pdm_68_aligned_menpo.txt': None
    },
    'patch_experts': {
        'ccnf_patches_0.25_general.txt': None,
        'ccnf_patches_0.25_wild.txt': None,
        'ccnf_patches_0.25_multi_pie.txt': None,
        'ccnf_patches_0.35_general.txt': None,
        'ccnf_patches_0.35_wild.txt': None,
        'ccnf_patches_0.35_multi_pie.txt': None,
        'ccnf_patches_0.5_general.txt': None,
        'ccnf_patches_0.5_wild.txt': None,
        'ccnf_patches_0.5_multi_pie.txt': None,        
        'ccnf_patches_1_wild.txt': None,
        'svr_patches_0.25_general.txt': None,
        'svr_patches_0.25_wild.txt': None,
        'svr_patches_0.35_general.txt': None,
        'svr_patches_0.35_wild.txt': None,
        'svr_patches_0.5_general.txt': None,
        'svr_patches_0.5_wild.txt': None
    },
    'model_inner': {
        'clnf_inner.txt': None,
        'main_clnf_inner.txt': None,
        'pdms': {'pdm_51_inner.txt': None},
        'patch_experts': {'ccnf_patches_1.00_inner.txt': None}
    },
    'model_eye': {
        'clnf_left_synth.txt': None,
        'clnf_right_synth.txt': None,
        'main_clnf_synth_left.txt': None,
        'main_clnf_synth_right.txt': None,
        'patch_experts': {
            'ccnf_patches_1.00_synth_lid_.txt': None,
            'ccnf_patches_1.50_synth_lid_.txt': None,            
            'left_ccnf_patches_1.00_synth_lid_.txt': None,
            'left_ccnf_patches_1.50_synth_lid_.txt': None
        },
        'pdms': {
            'pdm_28_l_eye_3D_closed.txt': None,
            'pdm_28_eye_3D_closed.txt': None
        }
    },
    'cen_general.txt': None,
    'clm_general.txt': None,
    'clm_wild.txt': None,
    'clnf_general.txt': None,
    'clnf_wild.txt': None,
    'clnf_multi_pie.txt': None,
    'main_ceclm_general.txt': None,
    'main_clm_general.txt': None,
    'main_clm_wild.txt': None,
    'main_clnf_demos.txt': None,
    'main_clnf_general.txt': None,
    'main_clnf_wild.txt': None,
    'main_clnf_multi_pie.txt': None,
    'early_term_cen_of.txt': None,
    'haarAlign.txt': None,
    'tris_68.txt': None,
    'tris_68_full.txt': None    
}


# CEN general model patches to download from Dropbox/OneDrive

onedrive_base = 'https://onedrive.live.com/download?cid=2E2ADA578BFF6E6E&'

patch_seq = ['0.25', '0.35', '0.50', '1.00']
cen_patches = {
    '0.25': {
        'main': 'https://www.dropbox.com/s/7na5qsjzz8yfoer/cen_patches_0.25_of.dat?dl=1',
        'backup': onedrive_base + 'resid=2E2ADA578BFF6E6E%2153072&authkey=AKqoZtcN0PSIZH4'
    },
    '0.35': {
        'main': 'https://www.dropbox.com/s/k7bj804cyiu474t/cen_patches_0.35_of.dat?dl=1',
        'backup': onedrive_base + 'resid=2E2ADA578BFF6E6E%2153079&authkey=ANpDR1n3ckL_0gs'
    },
    '0.50': {
        'main': 'https://www.dropbox.com/s/ixt4vkbmxgab1iu/cen_patches_0.50_of.dat?dl=1',
        'backup': onedrive_base + 'resid=2E2ADA578BFF6E6E%2153074&authkey=AGi-e30AfRc_zvs'
    },
    '1.00': {
        'main': 'https://www.dropbox.com/s/2t5t1sdpshzfhpj/cen_patches_1.00_of.dat?dl=1',
        'backup': onedrive_base + 'resid=2E2ADA578BFF6E6E%2153070&authkey=AD6KjtYipphwBPc'
    }
}


# Utility functions for downloading files

def _getinput(*args, **kwargs):
    # Python-agnostic function for getting console input.
    try:
        return raw_input(*args, **kwargs)
    except NameError:
        return input(*args, **kwargs)


def _download_file(f, parent, url):
    
    print("* Downloading '{0}'...".format(f))
    filepath = os.path.join(parent, f)
    try:
        mod_http = urlopen(url)
    except (URLError, HTTPError):
        raise RuntimeError("Failed to download '{0}'".format(url))
    with open(filepath, 'wb') as out:
        out.write(mod_http.read())


def _download_dir(d, root, path = []):
    
    dirpath = os.path.join(root, *path)
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
        
    for f in d.keys():
        # If file, download to current path
        if not d[f]:
            mod_url_suffix = "/".join(path) + "/" + f if len(path) else f
            mod_url = repo_base + base_modpath + mod_url_suffix
            _download_file(f, dirpath, mod_url)
        else:
            _download_dir(d[f], root, path = path + [f])


# Main function for downloading all OpenFace models

def download_models(modpath = None):
    """Downloads all models required to run OpenFace to a given local
    directory, including the large externally-hosted CEN patch files.
    
    If no path is provided, the current working directory will be used.
    """
    if not modpath:
        modpath = os.path.join(os.getcwd(), 'openface_models')
    else:
        parentdir = os.path.normpath(os.path.join(modpath, os.pardir))
        if not os.path.isdir(parentdir):
            raise RuntimeError("Path '{0}' does not exist".format(parentdir))
    
    print("\nAll OpenFace models will be downloaded to '{0}'\n".format(modpath))
    prompt = "This will take approximately 470 MB of disk space. Continue?"
    resp = _getinput(prompt + " (y/n): ")
    if not len(resp) or resp[0].lower() != 'y':
        print('')
        return
    
    # Create local model directory if it doesn't already exist
    if os.path.isdir(modpath):
        prompt = "\nDirectory '{0}' already exists. Overwrite? (y/n): "
        resp = _getinput(prompt.format(modpath))
        if not len(resp) or resp[0].lower() != 'y':
            print('')
            return
        else:
            shutil.rmtree(modpath)
    os.mkdir(modpath)
        
    print('\n=== Downloading OpenFace models ===\n')
    print('Destination: {0}\n'.format(modpath))
        
    # First, download smaller models from GitHub        
    _download_dir(model_files, modpath, path = ['model'])
    
    # Then, download OpenCV haarcascade classifier from GitHub
    classifier_url = repo_base + classifier_path + classifier_file
    classifier_dir = os.path.join(modpath, 'classifiers')
    if not os.path.isdir(classifier_dir):
        os.mkdir(classifier_dir)
    _download_file(classifier_file, classifier_dir, classifier_url)
    
    # Finally, download CEN patches to the model directory
    patch_dir = os.path.join(modpath, 'model', 'patch_experts')
    for patch in patch_seq:
        patch_name = 'cen_patches_{0}_of.dat'.format(patch)
        try:
            patch_url = cen_patches[patch]['main']
            _download_file(patch_name, patch_dir, patch_url)
        except RuntimeError:
            patch_url = cen_patches[patch]['backup']
            _download_file(patch_name, patch_dir, patch_url)
            
    print('\n=== Downloaded all OpenFace models sucessfully! ===\n')
    

def install_models():
    """Installs the full set of OpenFace models to a hidden folder in the user's
    home directory, allowing use of the OpenFace Python bindings from any folder.
    """
    homedir = os.path.expanduser("~")
    modpath = os.path.join(homedir, '.openface')
    download_models(modpath)
    
    
def local_mod_path():
    modpath = os.path.join(os.getcwd(), 'openface_models')
    if os.path.isdir(modpath):
        return modpath
    return None
    
    
def installed_mod_path():
    modpath = os.path.join(os.path.expanduser("~"), '.openface')
    if os.path.isdir(modpath):
        return modpath
    return None


def get_mod_path():
    modpath = local_mod_path()
    if not modpath:
        modpath = installed_mod_path()
    if not modpath:
        e = ("Could not find a useable model directory. Please run either "
            "'download_models()' to install the OpenFace models in the current "
            "folder, or 'install_models()' to install them globally for the "
            "current user."
        )
        raise RuntimeError(e)
    return modpath


if __name__ == '__main__':
    download_models()
