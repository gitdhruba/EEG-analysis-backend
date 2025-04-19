from flask import Flask, request, jsonify # type: ignore
from flask_cors import CORS # type: ignore
from dotenv import load_dotenv # type: ignore
from database import initialize_db
from routes import router
from eeg_processor import load_model
import os





if __name__ == "__main__":
    # load env variables
    load_dotenv()

    # create uploads directory
    os.makedirs(os.getenv("UPLOAD_DIR"), exist_ok=True)

    # load model
    load_model()
    
    # create flask server
    app = Flask(__name__)
    CORS( 
            app,
            origins="*",
            methods=["GET", "POST"]
        )
            
    # db initialization
    initialize_db(app)

    # add routes
    app.register_blueprint(router)

    # start
    app.run(
                debug=True,
                host="0.0.0.0"
           )