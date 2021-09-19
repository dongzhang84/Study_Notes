# Notes on Docker and Data Science Application

- Docker Introduction (installation, build an image, start a container, basic commands)
- First Python Application
- First C++ Application
- First Machine Learning Application
- Flask Basic
- First ML deployment using Flask
- Deploy ML model using Flask in Docker





## Docker Introduction

### Installation

Download Docker Desktop for Mac (see the [link](https://docs.docker.com/get-started/)). One can manage Docker images and containers on one's machine by **Docker Dashboard**.

![Docker_dashboard.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Docker/Docker_dashboard.png?raw=true)

 

### Build an Image

A Docker image is a file used to execute code in a Docker container. Docker images act as a set of instructions to build a Docker container, like a template. 

Here is the simplest way to build an image:

```
$ docker build -t IMAGE_NAME:TAG .
```

For example, in the [instruction](https://docs.docker.com/get-started/02_our_app/) to build the "hello world" image by (without tag):

```
$ docker build -t getting-started .
```

the `-t` flag tags our image. Think of this simply as a human-readable name for the final image. Since we named the image `getting-started`, we can refer to that image when we run a container.

The `.` at the end of the `docker build` command tells that Docker should look for the `Dockerfile` in the current directory.



### Start a Container

A container is a standard unit of software that packages up code and all its dependencies so the application runs quickly and reliably from one computing environment to another. See the demo picture for the architecture:

![img](https://www.docker.com/sites/default/files/d8/styles/large/public/2018-11/container-what-is-container.png?itok=vle7kjDj)

The command to start a container the simplest way:

```
$ docker run IMAGE_NAME:TAG
```

 

### Basic Commands

- We can view containers in Docker Dashboard

- How to check containers? 

  ```
  $ docker ps
  ```

  or (also see stoped containers)

  ```
  $ docker ps -a
  ```

- Stop and remove docker container

  ```
  $ docker stop <the-container-id>
  $ docker rm <the-container-id>
  ```

  Or stop and remove a container in a single command (or do it from the Dashboard):

  ```
  $ docker rm -f <the-container-id>
  ```

- Check all images

  ```
  $ docker images
  ```

- Remove an image (or do it from the Dashboard)

  ```
  $ docker image rm <the-image-id>
  ```

  

## First Python Application

The first Python application only includes two files in the directory:

- hello.py
- Dockerfile

In **hello.py**, just put one line: 

```python
print("Hello from Docker container :)")
```

In **Dockerfile**, put

```
FROM python:3.7

WORKDIR /app

COPY hello.py .

CMD ["python", "./hello.py"]
```

Note than an other version gives (they are basically the same)

```
FROM python:3.7

WORKDIR /usr/src/app

COPY hello.py .

CMD ["python", "./hello.py"]
```



One can test the python code without Docker by

```
$ python3 hello.py
```

Or build an image and start a container by:

```
$ docker build -t hello_world .
$ docker run hello_world
```

In the terminal one can observe the following result:

![first_python.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Docker/first_python.png?raw=true)

Check the container (which has been stoped) by 

```
docker ps -a
```

![first_python_check_container.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Docker/first_python_check_container.png?raw=true)

You can remove the container by the command mentioned in the previous section. 

For running a container, one can also use

```
$ docker run --rm hello_world
```

So the container will be removed instantaneously right after running the python code. 

Discussion about all flag commands can be seen in [this link](https://docs.docker.com/engine/reference/run/).



## First C++ Application

In a directory write up a C++ file called **main.cpp**:

```c++
# include <iostream>

int main(int argc, char const *argv[])
{
    std::cout <<"Hello Docker container!"<< std::endl;
    return 0;
}
```

And in the command line you can type:

```
$ g++ -o test main.cpp
```

to generate a compiler called **test**. 

Now run it:

```
$ ./test
```

The result is shown in the terminal: 

```
$ Hello Docker container!
```

![first_cpp_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Docker/first_cpp_1.png?raw=true)



Now create a **Dockerfile**:

```
FROM gcc:latest

COPY . /usr/src/cpp_test

WORKDIR /usr/src/cpp_test

RUN g++ -o test main.cpp

CMD ["./test"]
```

and build an image and start a container by

```
$ docker build -t cpp_test .
$ docker run --rm cpp_test
```

The result is as follows:

![first_cpp_2.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Docker/first_cpp_2.png?raw=true)





## First Machine Learning Application

We look at the [iris flower problem](https://www.kaggle.com/arshid/iris-flower-dataset). This is a very basic classification problem. 

In a directory I put the following files: 

- iris.csv
- train.py
- predict.py

The **iris.csv** is the dataset. The **train.py** file trains and output the model: 

```python
import pandas as pd
import joblib

from sklearn import svm 
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load data
df = pd.read_csv('iris.csv')


# dataset spliting    
X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df.Species

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 24)

print("size of X_train: " ,X_train.shape)
print("size of y_train: " ,y_train.shape)
print("size of X_test: " ,X_test.shape)
print("size of y_test: " ,y_test.shape)

model = svm.SVC() # select the svm algorithm

# we train the algorithm with training data and training output
model.fit(X_train, y_train)

# we pass the testing data to the stored algorithm to predict the outcome
y_pred = model.predict(X_test)
print('The accuracy of the SVM is: ', metrics.accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred))

joblib.dump(model, "model.pkl")
```

And the **predict.py** file test the prediction pipeline:

```python
import joblib

input = [5.8,2.8,5.1,2.4]
model_load = joblib.load('model.pkl')
prediction = model_load.predict([input])

print("Prediction: ", prediction)
```

Without Docker one can use the command line

```
$ python3 train.py
```

to output a model pickle file called **model.pkl**, and

```
$ python3 predict.py
```

 to show a prediction for  input = [5.8,2.8,5.1,2.4]: 

```
Prediction:  ['Iris-virginica']
```



The Dockerfile is written as 

```dockerfile
FROM python:3.7

WORKDIR /app

COPY iris.csv ./iris.csv
COPY train.py ./train.py
COPY predict.py ./predict.py

RUN pip install pandas scikit-learn joblib
RUN python train.py

CMD ["python", "predict.py"]
```

or 

```dockerfile
FROM python:3.7

COPY . /app

WORKDIR /app

RUN pip install pandas scikit-learn joblib
RUN python train.py

CMD ["python", "predict.py"]
```

Then build the image by

```
$ docker build -t first_ml .
```

(Note that the image runs to the line before "CMD"). To create a container and run the predictor:

```
$ docker run first_ml
```

![first_ml_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Docker/first_ml_1.png?raw=true)

 



## Flask Basic

A very good instruction can be seen in [this link](https://medium.com/@nutanbhogendrasharma/deploy-machine-learning-model-with-flask-on-heroku-cd079b692b1d). Recall python flask. Create a python file **app.py** as 

```python
#import Flask 
from flask import Flask

#create an instance of Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"
    
if __name__ == '__main__':
    app.run(debug=True)
```

Run **app.py** (the flask application) by 

```
$ python3 app.py
```

This launches a very simple builtin server http://127.0.0.1:5000/:

![flask_hello_world.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Docker/flask_hello_world.png?raw=true)



## Model Deployment with Flask

Deploy the **iris flower model** with Flask. One example can be seen in [this blog](https://medium.com/@nutanbhogendrasharma/deploy-machine-learning-model-with-flask-on-heroku-cd079b692b1d). Here I give a simpler version, based on the description in [another blog](https://medium.com/analytics-vidhya/deploying-a-machine-learning-model-using-flask-for-beginners-674944714b86).

Create a folder with the following files:

![flask_ml_2.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Docker/flask_ml_2.png?raw=true)

Here the **train.py** is necessary to generate the model pickle file. The **index.html** and **result.html** are two basic html pages to load the input page and prediction (output) page. 

The **index.html** is written as

```html
<html> 
<body> 
    <h3>Iris Flowers Classification</h3> 
  
<div> 
  <form action="/result" method="POST"> 
    <label for="Sepal Length">Sepal Length</label> 
    <input type="text" id="Sepal Length" name="Sepal Length"> 
    <br>

    <label for="Sepal Width">Sepal Width</label> 
    <input type="text" id="Sepal Width" name="Sepal Width"> 
    <br> 
    
    <label for="Petal Length">Petal Length</label> 
    <input type="text" id="Petal Length" name="Petal Length"> 
    <br>

    <label for="Petal Width">Petal Width</label> 
    <input type="text" id="Petal Width" name="Petal Width"> 
    <br>

    <input type="submit" value="Submit"> 
  </form> 
</div> 
</body> 
</html>
```

 which is used to input features for prediction.

The **result.html** page is written as:

```html
<!doctype html>
<html>
<body>
<h3> Prediction: {{prediction}}</h3>
</body>
</html>
```



More importantly, the **app.py** is the Flask python file to deploy the model. The above section gives the simplest application file. And here the **app.py** is modified to a more complicated version as follows:

```python
import joblib

#create an instance of Flask
app = Flask(__name__)

@app.route('/')
@app.route('/index')

def home():
    return render_template('index.html')

@app.route('/result', methods = ['GET','POST'])
def result():
    if request.method == 'POST':
       input_list = request.form.values()
       input_list = list(map(float, input_list))
       print(input_list) # test the input

       model_load = joblib.load('model.pkl') # load the model
       prediction = model_load.predict([input_list])[0] # do the prediction

       return render_template('result.html', prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True)
```



To run the application, one needs to first run

```
$ python3 train.py
```

to generate the model pickle file **model.pkl**. Then to run the Flask file

```
$ python3 app.py
```

![flask_ml_app.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Docker/flask_ml_app.png?raw=true)

Open the local server http://127.0.0.1:5000/: 

![flask_ml_input.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Docker/flask_ml_input.png?raw=true)

Let us input values: 

(Sepal Length) 5.8

(Sepal Width) 2.8

(Petal Length) 5.1

(Petal Width) 2.4

Click the "Submit" button, the output prediction is

![flask_ml_output.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Docker/flask_ml_output.png?raw=true)So the basic Flask application is running successfully. Note that in the terminal you can see something like:

![flask_ml_terminal.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Docker/flask_ml_terminal.png?raw=true)



## Deploy ML model using Flask in Docker

Now I would like to deploy the ML model with Flask and Docker. 

Here is the all files one needs for model deployment:

![flask_Docker_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Docker/flask_Docker_1.png?raw=true)

The **requirements.txt** list the libraries should be installed (I do not specify the package versions):

```
pandas
joblib
scikit-learn
Flask
```



Note that the **app.py** file for the Docker version is slightly different from the local version. The last line is changed from 

```python
if __name__ == '__main__':
    app.run(debug=True)
```

to

```python
if __name__ == '__main__':
    app.run(host='0.0.0.0')
```

 

The **Dockerfile** is:

```dockerfile
FROM python:latest

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

RUN python train.py

CMD ["python", "app.py"]
```

Now launch a Docker image by

```
$ docker build -t flask_ml .
```

Once the image is built, run the application by

```
$ docker run -p 5000:5000 flask_ml
```

Note that it is different from the local version. The server link to open the app is

http://0.0.0.0:5000/: 

![flask_Docker_input.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Docker/flask_Docker_input.png?raw=true)

Then it will be perfect to run the application. Here is the terminal screenshot:

![flask_Docker_terminal.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/Docker/flask_Docker_terminal.png?raw=true)



The relevant code can be found here:

https://github.com/dongzhang84/Docker_tutorial



