from flask import Blueprint, request, jsonify # type: ignore
from werkzeug.utils import secure_filename # type: ignore
from models import Subject, EI, PSD
from sqlalchemy.orm import joinedload
from database import db
from eeg_processor import bands, event_list, process_eeg_file
import os

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
        
    # subject name
    subject_name : str = eeg_file.filename.split(".")[0].strip()

    # subject type
    subject_type : int = request.form.get("type", -1, type=int)
    if subject_type not in [0, 1]:
        return jsonify({"error": "type should be given and it should be either 0(for VIP) or 1(for normal)"}), 400
    
    # event desc
    event_desc : str = request.form.get("event_desc")
    if event_desc == None or event_desc == "":
        return jsonify({"error": "event_desc should be given"}), 400

    event_desc = event_desc.strip().capitalize()
    if event_desc not in event_list:
        return jsonify({"error": "unknown event_desc"}), 400
    
    # check for type inconsistency, if it is there then update or create new subject
    subject : Subject = db.session.query(Subject).filter(Subject.name == subject_name).first()
    subject_id : int = -1
    if subject:
        subject.type = subject_type
        db.session.commit()
        subject_id = subject.id
    else:
        new_subject : Subject = Subject(subject_name, subject_type)
        db.session.add(new_subject)
        db.session.commit()
        subject_id = new_subject.id
    
    # from now onwards we have subject_id


    upload_dir : str = os.getenv("UPLOAD_DIR")
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
        
        # get result from eeg_processor
        result, psds, cleaned_data, ei_val = process_eeg_file(vhdr_filepath, vmrk_filepath, event_desc)


        if ei_val is not None and psds is not None:
            # save ei
            ei : EI = db.session.query(EI).filter((EI.subject_id == subject_id) & (EI.event == event_desc)).first()
            if ei:                  # update
                ei.value = float(ei_val)
            else:                   # create
                new_ei = EI(subject_id, event_desc, float(ei_val))
                db.session.add(new_ei)

            # save psds
            for band in psds:
                freq : list[float] = [float(it[0]) for it in band["points"]]
                pxx : list[float] = [float(it[1]) for it in band["points"]]

                psd : PSD = db.session.query(PSD).filter((PSD.subject_id == subject_id) & (PSD.event == event_desc) & (PSD.band == band["band"])).first()
                if psd:             # update
                    psd.frequencies = freq
                    psd.pxx_values = pxx
                else:               # create
                    new_psd : PSD = PSD(subject_id, event_desc, band["band"], freq, pxx)
                    db.session.add(new_psd)

            db.session.commit()

        # close all brain-Vision files
        vmrk_file.close()
        vhdr_file.close()
        eeg_file.close()
        # remove them
        os.remove(vmrk_filepath)
        os.remove(vhdr_filepath)
        os.remove(eeg_filepath)

        return jsonify({
            "result": result,
            "plot_data": psds,
            "engagement_index": ei_val,
            "cleaned_data": cleaned_data
        }), 200


    else:
        return jsonify({"error": "File upload failed"}), 500
    




@router.route('/get-saved-data', methods=['GET'])
def get_saved_data():

    # data-structure
    '''
        {
            subjects: [subject_names]
            eis: [  # index = subject_index
                    [(event, value)]
                 ],

            psds: [  
                     {
                        "band" : band,
                        "points" : [   # index = subject_index
                                      {
                                         "event": [(freq, pxx)]
                                      }
                                   ]
                     }
                  ]
        }
    '''

    # prepare ds for subjects
    subjects : list[Subject] = (
                                    db.session.query(Subject)
                                        .options(
                                            joinedload(Subject.eis),
                                            joinedload(Subject.psds)
                                        ).all()
                               )    
    
    eis : list[list[tuple[str, float]]] = [
                                                [
                                                    (ei.event, ei.value) for ei in sub.eis
                                                ]  
                                                for sub in subjects
                                              ]

    psds : list[dict] = [{"band": b[0], "points": [dict() for _ in range(len(subjects))]} for b in bands]
    for sub_idx, sub in enumerate(subjects):
        for psd in sub.psds:
            band_idx = next((i for i, t in enumerate(bands) if t[0] == psd.band), -1)
            psds[band_idx]["points"][sub_idx][psd.event] = list(zip(psd.frequencies, psd.pxx_values))




    return jsonify({
        "data": {
                    "subjects": [(sub.name, sub.type) for sub in subjects],
                    "eis": eis,
                    "psds": psds,
                }
    }), 200