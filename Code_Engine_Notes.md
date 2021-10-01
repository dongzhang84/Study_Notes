# Code Engine Notes





## 1. Installation

First of all, this is the Code Engine console

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





## 2. Create Hello World Project with App and Jobs

Follow [this link](https://cloud.ibm.com/docs/codeengine?topic=codeengine-manage-project).



### 2. 1 Create A Project

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



### 2.2 Deploy an application from the console

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

### 2.3 Deploy an application from the CLI terminal

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



### 2.4 Run an app from images in Container Registry

Follow [this link](https://cloud.ibm.com/docs/codeengine?topic=codeengine-deploy-app-crimage):





