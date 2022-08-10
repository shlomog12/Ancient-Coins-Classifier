from flask import Flask, request, jsonify
from flask_cors import CORS
from train import train_model, resize_image, remove_cache

app = Flask(__name__)
CORS(app)

@app.route('/', methods= ['GET'])
def get_message():
    print("Got request in main function")
    return 'Zuzim server version 1.0.0', 200

@app.route('/api/run_model', methods=['POST'])
def run_model():
    print("Got request in static files")
    print(request.files)
    f = request.files['static_file']
    new_img = resize_image(f)
    resModel = train_model(new_img)
    resp = {"success": True, "response": "file saved!", "resModel": resModel}
    remove_cache()
    return jsonify(resp), 200
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3555, debug=True)



