Description of the contents of local directory:

local
    |---transformer ==> This folder is the replica of espnet/nets/pytorch_backend/transformer with a few additonal files for DANN.
        |---encoder_dann.py ==> This is a custom encoder with the option to specify no.of encoder layers to share.     
    
    acc_classifier.py ==> Classifier model classes. Has both simple MLP and LSTM definitions.
    accent_embeddings.py ==> This code extracts and saves embeddings from a pre-trained accent classifier.
    add_labels.py ==> The base IITM data has no accent labels. I use this code to add the accent label info to json. Not required post KMeans.
    clf_tr.py ==> Train an accent classifier using ASR-transformer embeddings.
    dann_asr_recog.py ==> Replica of espnet/asr/pytorch_backend/recog.py
    dann_asr_train.py ==> Calls the main training code, and a few arguments. Acts as a wrapper code.
    dann_asr.py ==> The main training code.
    DANN_KMeans.py ==> The DANN model class that takes E2E and Classifier models as inputs.
    DANN.py ==> Same as DANN_KMeans. Just added this in case we need to tinker a lot and we do not break the main DANN_KMeans code.
    filtering_samples.py ==> Comes with the wsj recipe by default.
    find_transcripts.pl ==> Comes with the wsj recipe by default.
    flist2scp.pl ==> Comes with the wsj recipe by default.
    io_utils_dann_rit.py ==> The input output utility function. Here we add the accent labels to inputs.
    kmeans.py ==> The code to perform KMeans clustering of accent embeddings and save the corresponding cluster labels to data.json.
    load_tr_model.py ==> Code to load the pre trained DANN model.
    ndx2flist.pl ==> Comes with the wsj recipe by default.
    normalize_transcript.pl ==> Comes with the wsj recipe by default. 

