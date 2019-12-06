import time
import flask
import numpy as np
from PIL import Image, ExifTags
from keras import backend as K
from keras.models import load_model

app = flask.Flask(__name__)


def load():
    """modelを読み込む"""
    model = load_model("mnist_cnn_model.h5", compile=False)
    return model


def transform_img(img):
    """読み込んだimageのshapeをMNISTのshape(28, 28)にする"""
    img = img.convert('L')
    width,height = 28, 28
    img = img.resize((width,height), Image.LANCZOS)
    img_array = np.asarray(img).reshape((1, width, height, 1))
    return img_array


@app.route("/", methods=["POST"])
def predict():
    start = time.time()
    model = load()
    response = {"Content-Type": "application/json",
                "result": None}

    if flask.request.method == "POST":
        if flask.request.files["file"]:
            img = Image.open(flask.request.files["file"])
            img_array = transform_img(img)
            result = model.predict(img_array,verbose=0)
            K.clear_session()
            response["result"] = str(np.argmax(result))

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    return flask.jsonify(response)


if __name__ == '__main__':
    app.run(threaded=False)