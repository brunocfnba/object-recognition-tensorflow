## How to run the code
This is an API created that uses the object detection model created by Google.

* Make sure you installed all the required libs (found in the requirements.txt file).
* Run `python start.py`, it will run on port 8080 on your localhost.
>Ie created a stub.html file so you can upload images to be analyzed by the model and check its result.

### How the code works

* The 'start.py' exposes an API endpoint using Flask that handles the image and call the function responsible to analyze the image.
* The 'main.py' file is where the model is loaded using TensorFlow, the image sent is prepared and transformed into the format expected by the model and process the returned info that is then sent back to the API handler.

For more details refer to [Google's repo](https://github.com/tensorflow/models/tree/master/object_detection)
