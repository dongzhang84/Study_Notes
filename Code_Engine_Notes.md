# Code Engine Notes

- Installation
- Create hello world project
- Deploy hello world app (using public registry)
- Create and run hello world job (including how to create myregistry)
- How to push Container image from local to IBM Container Registery 
- Summary (steps)
- Case Study: First ML Job
- Case Study: First ML app



## 1. Installation

First of all, this is the Code Engine console

https://cloud.ibm.com/codeengine/overview

![console.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/code_engine/console.png?raw=true)



#### Step 1: Install IBM Cloud CLI

Follw [this link](https://cloud.ibm.com/docs/cli?topic=cli-getting-started): Install it locally by

```
$ curl -sL https://raw.githubusercontent.com/IBM-Cloud/ibm-cloud-developer-tools/master/linux-installer/idt-installer | bash
```

Verify the installation

```
$ ibmcloud dev help
```



#### Step 2: Configure your environment

Login to IBM cloud with your IBMid by

```
$ ibmcloud login
```

But for security in most cases you need to do

```
$ ibmcloud login --sso
```



####  Step 3: Set up the CLI

Follow [this link](https://cloud.ibm.com/docs/codeengine?topic=codeengine-install-cli): Once you login to the IBM Cloud CLI, you must specify a resource group. To get the list of your resource groups, you can do:

```
$ ibmcloud resource groups
```



#### Step 4: Instal the Code Engine CLI

Use ibmcloud to do it:

```
$ ibmcloud plugin install code-engine
```

and check if it is installed by

```
$ ibmcloud plugin show code-engine
```





## 2. Create Hello World Project

Follow [this link](https://cloud.ibm.com/docs/codeengine?topic=codeengine-manage-project).



There are two ways to create project. 

From the [Project page on the Code Engine console](https://cloud.ibm.com/codeengine/create/project), you can create it:

![create_project.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/code_engine/create_project.png?raw=true)

Note that you need to select location, and select resource group. 



Another way to create project is from the Code Engine CLI terminal, use a project name that is unique to your region:

```
$ ibmcloud ce project create --name PROJECT_NAME 
```

Note that before doing the above command you have already selected your location and resource group. 

Verify your new project is created by 

```
$ ibmcloud ce project get --name PROJECT_NAME
```



**List all projects**

You can see all projects by

```
$ ibmcloud ce project list
```



**Select a project**

```
$ ibmcloud ce project select --name PROJECT_NAME
```



**Delete a project**

Soft delete:

```
$ ibmcloud ce project delete --name myproject -f
```

Hard delete:

```
$ ibmcloud ce project delete --name myproject1 --hard -f
```



## 3. Deploy an Application



### 3.1 Deploy an application from the console

From the console:

![deploy_app_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/code_engine/deploy_app_1.png?raw=true)

There are two ways to deploy an app: (1) Run a container image. (2) Star with source code. 

Selec tto run a container image, you will go to this page:

![deploy_app_2.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/code_engine/deploy_app_2.png?raw=true)

Name the application as **myapp**, note that you need to select a project (for example, the created "hello_world"), and chose a **container image**, or from **source code**.

For simplicity, you can select a public container image like the following: 

```
docker.io/ibmcom/codeengine
```

 Or do the **"Configure image"**:

![deploy_app_3.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/code_engine/deploy_app_3.png?raw=true)



The above is for public container image. 

Once you create the Application, you will have an app like this:

![deploy_app_4.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/code_engine/deploy_app_4.png?raw=true)

Click the **"Open applicaiton URL"** on the top right of the above screenshot, you will have the application result like this:

![deploy_app_5.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/code_engine/deploy_app_5.png?raw=true)

### 3.2 Deploy an application from the CLI terminal

Follow [this link](https://cloud.ibm.com/docs/codeengine?topic=codeengine-deploy-app-tutorial):

Select resource --> select project

Then deploy the hello world app by:

```
$ ibmcloud ce application create --name myapp --image docker.io/ibmcom/codeengine
```

Once the app is deployed, check the details of the app by

```
$ ibmcloud ce application get -n myapp
```

To open the app, check the url:

```
$ ibmcloud ce application get -n myapp -output url
```

you will get

```
https://myapp.epf8v6uxdpw.us-south.codeengine.appdomain.cloud
```

Then copy past the url:

```
$ curl https://myapp.epf8v6uxdpw.us-south.codeengine.appdomain.cloud
```

You can get the output:

```
Hello World from:
. ___  __  ____  ____
./ __)/  \(    \(  __)
( (__(  O )) D ( ) _)
.\___)\__/(____/(____)
.____  __ _   ___  __  __ _  ____
(  __)(  ( \ / __)(  )(  ( \(  __)
.) _) /    /( (_ \ )( /    / ) _)
(____)\_)__) \___/(__)\_)__)(____)

Some Env Vars:
--------------
CE_APP=myapp
CE_DOMAIN=us-south.codeengine.appdomain.cloud
CE_SUBDOMAIN=epf8v6uxdpw
HOME=/root
HOSTNAME=myapp-00001-deployment-69c4b6bb8c-8pwgh
K_REVISION=myapp-00001
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PORT=8080
PWD=/
SHLVL=1
z=Set env var 'SHOW' to see all variables
```



### 3.3 Deploy an app from images in Container Registry

Follow [this link](https://cloud.ibm.com/docs/codeengine?topic=codeengine-deploy-app-crimage):





## 4. Run a Job

### 4.1 Run a job from public container

**Create and Submit a job from the console**

![run_job_1.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/code_engine/run_job_1.png?raw=true)

- Job name: firstjob

- Select project: for this one select **hello_world** (the project just create)
- Container image from a public registry: docker.io/ibmcom/codeengine

Once you done, click **Create** and then you can **submit the job**. 



**Create and Submit a job from the CLI terminal**

See [this link](https://cloud.ibm.com/docs/codeengine?topic=codeengine-create-job)

```
$ ibmcloud ce job create --name firstjob --image docker.io/ibmcom/codeengine
```

Example output

```
Creating job 'myjob'...
OK
```

Check all jobs you have in the selected project:

```
$ ibmcloud ce job list
```



Run job from the CLI terminal:

```
$ ibmcloud ce jobrun submit --job firstjob
```

You can also add:

```
$ ibmcloud ce jobrun submit --job firstjob --array-indices "1 - 5" 
```

to specify array-indices (see details [here](https://cloud.ibm.com/docs/codeengine?topic=codeengine-run-job)).

You will have output like this:

```
Getting job 'firstjob'...
Submitting job run 'firstjob-jobrun-ni9xj'...
Run 'ibmcloud ce jobrun get -n firstjob-jobrun-ni9xj' to check the job run status.
OK
```

and you can check the job run status. The "wrong job" (for example, wrong image) will show:

```
Instances:    
  Name                        Running  Status   Restarts  Age  
  secondjob-jobrun-hu7bs-0-0  0/0      Pending  0  
```

and not able to be run. Otherwise it will show:

```
Instances:    
  Name                       Running  Status     Restarts  Age  
  firstjob-jobrun-ni9xj-0-0  0/1      Succeeded  0         4m16s  
```

on the bottom of the job status. 



**Accessing job details with the CLI**

```
$ ibmcloud ce job get --name firstjob
```

will show something like this:

```
Getting job 'firstjob'...
OK

Name:          firstjob
ID:            abcdefgh-abcd-abcd-abcd-1a2b3c4d5e6f
Project Name:  myproject
Project ID:    01234567-abcd-abcd-abcd-abcdabcd1111
Age:           2m4s
Created:       2021-02-17T15:41:12-05:00

Image:                ibmcom/firstjob
Resource Allocation:
    CPU:     1
    Memory:  4G

Runtime:
    Array Indices:       0
    Max Execution Time:  7200
    Retry Limit:         3
```



**Obviously, the console is easier than the CLI terminal to submit jobs and check job status.**



### 4.2 Run a job from images in Container Registry

- The first step is to **create an IAM API key** from here:

https://cloud.ibm.com/iam/apikeys

![API_key.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/code_engine/API_key.png?raw=true)

Save your API key. 



- The next step is to **add registry access to Code Engine**.

```
$ ibmcloud ce registry create --name myregistry --server us.icr.io --username iamapikey --password API_KEY
```

where API_KEY is the key you just created. 

The example output:

```
Creating image registry access secret 'myregistry'...
OK
```



And you can check you have the registry access:

![myregistry.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/code_engine/myregistry.png?raw=true)



- The third step is to **create your job configuration**. 

```
$ ibmcloud ce job create --name myhellojob --image us.icr.io/mynamespace/hello_repo --registry-secret myregistry
```

Note that here **us.icr.io/mynamespace/hello_repo** is just a demonstartion. You need to use a right image reference. The next session shows how to **push container images to IBM Cloud Container Registry**. 



Of course the console is better to select image and create job, like this:

![create_job_image.png](https://github.com/dongzhang84/Study_Notes/blob/main/figures/code_engine/create_job_image.png?raw=true)

Note that to configure image, you need to select:

- the right Registry server (e.g., us.icr.io)

- Registry access (e.g, myregistry, as just created)

- Namespace (for public it is ibmcom, next session shows how to select others)

- Repository image name (next session show you how to set up it so you can select for nex time)

- Tag: image tag, do not forget to select, the default is lattest

  

You can update a job change name and image follow [this link](https://cloud.ibm.com/docs/codeengine?topic=codeengine-update-job). But it is easier to update in the console. 

More details for create and running jobs see [this link and the following tutorials](https://cloud.ibm.com/docs/codeengine?topic=codeengine-create-job-crimage). 







## 5. How to use IBM Cloud Container Registry

See [this link for more details](https://cloud.ibm.com/docs/Registry?topic=Registry-getting-started#getting-started). 

- Step 1: **Set up a namespace**

```
$ ibmcloud login --sso
```

add namespace, you can check namespace by:

```
$ ibmcloud cr namespace-list -v
```

and add one namespace:

```
ibmcloud cr namespace-add <your_namespace>
```



- Step 2: Pull images from another registry to your local computer

For example, pull a docker image to your local:

```
$ docker pull hello-world:latest
```

and tage it:

```
$ docker tag hello-world:latest us.icr.io/<your_namespace>/hw_repo:1
```

The more general way to pull docker images and tag it:

```
$ docker pull <source_image>:<tag>
```

```
$ docker tag <source_image>:<tag> <region>.icr.io/<your_namespace>/<new_image_repo>:<new_tag>
```



- Step 3: Push Docker image to your namaspace

First login to IBM Cloud Container Registry:

```
$ ibmcloud cr login
```



For the above example, do 

```
$ docker push hello-world:latest us.icr.io/<your_namespace>/hw_repo:1
```

A more general command:

```
$ docker push <region>.icr.io/<your_namespace>/<image_repo>:<tag>
```

Now the image is pushed into IBM Container Registry. 

Go back to create a job:

```
$ ibmcloud ce job create --name myhellojob --image us.icr.io/<your_namespace>/hw_repo:1 --registry-secret myregistry
```

or more general

```
$ ibmcloud ce job create --name myhellojob --image <region>.icr.io/<your_namespace>/<image_repo>:<tag> --registry-secret myregistry
```

to use your own registry. 











## 5. Summarize Steps

1. **Login in ibmcloud**

   ```
   ibmcloud login --sso
   ```

2. **Select resource group**

   ```
   ibmcloud resource groups
   ```

   ```
   ibmcloud target -g RESOURCE_GROUP
   ```

3. **Select project (or create project)**

   ```
   ibmcloud ce project select --name PROJECT_NAME
   ```

4. **push the container image to Docker/IBM registry**

    login to IBM Cloud Container Registry:

   ```
   ibmcloud cr login
   ```

   Note that for the first time you need to create your own registry access, follow the details in Session **4.2**. 

   ```
   docker tag <source_image>:<tag> <region>.icr.io/<your_namespace>/<new_image_repo>:<new_tag>
   ```

   The source image is in your local computer. Usually US people use **us.icr.io** as the location, check what namespace you have, and name a new image repo with a tag. 

   Then push it to the IBM Container registry:

   ```
   docker push <region>.icr.io/<your_namespace>/<image_repo>:<tag>
   ```



Then:

5. **Create a job**

   ```
   ibmcloud ce job create --name myhellojob --image us.icr.io/<your_namespace>/hw_repo:1 --registry-secret myregistry
   ```

   The "hello world" way (use public image):

   ```
   ibmcloud ce job create --name firstjob --image docker.io/ibmcom/codeengine
   ```

   

6. **Submit the job**

   ```
   ibmcloud ce jobrun submit --job firstjob
   ```

   Of course you can use the console, will be easier than the CLI terminal.



Or 

5. **Deploy an app**

   Similar to create/run job.

 



## 6. Case Study: First ML application

Remember the example here:

![img](https://raw.githubusercontent.com/dongzhang84/Study_Notes/main/figures/Docker/first_ml_1.png)

where the code is here:

https://github.com/dongzhang84/Docker_tutorial/tree/main/first_ML/test1



In order to push this ML job to Code Engine, you can do locally:

```
$ docker tag first_ml:1 us.icr.io/<your_namespace>/iris:1
```

where **us.icr.io/<your_namespace>/iris:1** gives the tag for later Code Engine Push. 

Then login to Code Engine by the following steps:

```
ibmcloud login --sso
```

select resource group:

```
ibmcloud target -g RESOURCE_GROUP
```

select project:

```
ibmcloud ce project select --name hello_world
```

login to IBM container registry:

```
ibmcloud cr login
```

Push the local image to IBM container registry:

```
docker push us.icr.io/<your_namespace>/iris:1
```



Now you can run the job by CLI terminal:

```
ibmcloud ce job create --name iris --image us.icr.io/<your_namespace>/iris:1 --registry-secret myregistry
```

```
ibmcloud ce jobrun submit --job iris
```

Or do it from the console. 







## 7. Case Study: First ML App (underway)

Recall the Iris Flask application with Docker:

https://github.com/dongzhang84/Docker_tutorial/tree/main/first_flask_ML/flask_ML_Docker

![flask_Docker_terminal.png](https://github.com/dongzhang84/Study_Notes/raw/main/figures/Docker/flask_Docker_terminal.png?raw=true)

The local command:

```
docker build -t flask_ml:1 .
docker run -p 5000:5000 flask_ml:1 
```

and re-tag the image as:

```
docker tag flask_ml:1 us.icr.io/<your_namespace>/iris_app:1
```



Now login to IBM cloud, select Resource Group, select Project. 

```
ibmcloud cr login
```

```
docker push us.icr.io/<your_namespace>/iris_app:1
```

Try to deploy the app by:

```
ibmcloud ce application create --name irisapp --image us.icr.io/<your_namespace>/iris_app:1 --registry-secret myregistry
```

or by the console UI. 



But neither are succesful. 



