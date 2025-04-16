from flask import Blueprint, request, jsonify # type: ignore
from werkzeug.utils import secure_filename # type: ignore
import os
import eeg_processor

router = Blueprint('main', __name__)


@router.route('/')
def hello():
    return jsonify({"text": "hello"}), 200


@router.route('/predict', methods=['POST'])
def predict():
    # 3 files are required .eeg, .vmrk, .vhdr
    for file in ["eeg", "vmrk", "vhdr"]:
        if file not in request.files:
            return jsonify({f"error": "No {file} in the request"}), 400

    eeg_file = request.files['eeg']
    vmrk_file = request.files['vmrk']
    vhdr_file = request.files['vhdr']

    if eeg_file.filename == '' or vmrk_file.filename == '' or vhdr_file.filename == '':
        return jsonify({"error": "eeg, vmrk, vhdr all of them should be selected"}), 400
    
    # check if all filename are same except extension
    if eeg_file.filename.split(".")[0] != vmrk_file.filename.split(".")[0] or vmrk_file.filename.split(".")[0] != vhdr_file.filename.split(".")[0]:
        return jsonify({"error": "All file names should be same except extension"}), 400
        
    # event desc
    event_desc = request.form.get("event_desc")
    if event_desc == None or event_desc == "":
        return jsonify({"error": "event_desc should be given"}), 400

    upload_dir = os.getenv("UPLOAD_DIR")
    if eeg_file and vmrk_file and vhdr_file:
        # save eeg
        eeg_filename = secure_filename(eeg_file.filename)
        eeg_filepath = os.path.join(upload_dir, eeg_filename)
        eeg_file.save(eeg_filepath)
        
        # save vmrk
        vmrk_filename = secure_filename(vmrk_file.filename)
        vmrk_filepath = os.path.join(upload_dir, vmrk_filename)
        vmrk_file.save(vmrk_filepath)
        
        # save vhdr
        vhdr_filename = secure_filename(vhdr_file.filename)
        vhdr_filepath = os.path.join(upload_dir, vhdr_filename)
        vhdr_file.save(vhdr_filepath)
        
        # get result from eeg
        # result, pxx_avg, band_frequencies = process_eeg_file(vhdr_filepath, vmrk_filepath, event_desc)

        result, plot_data, cleaned_data, EI = eeg_processor.process_eeg_file(vhdr_filepath, vmrk_filepath, event_desc)

    

        return jsonify({
            "result": result,
            "plot_data": plot_data,
            "engagement_index": EI,
            "cleaned_data": cleaned_data
        }), 200


    else:
        return jsonify({"error": "File upload failed"}), 500