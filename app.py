from flask import Flask, request, jsonify # type: ignore
from flask_cors import CORS # type: ignore
from dotenv import load_dotenv # type: ignore
import os
import database
import routes



if __name__ == "__main__":
    # load env variables
    load_dotenv()

    # create uploads directory
    os.makedirs(os.getenv("UPLOAD_DIR"), exist_ok=True)

    # create flask server
    app = Flask(__name__)
    CORS(app)
            
    # db initialization
    database.initialize_db(app)

    # add routes
    app.register_blueprint(routes.router)

    # start
    app.run(debug=True)