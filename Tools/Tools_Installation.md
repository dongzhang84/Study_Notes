# Tool Installation



## Anaconda

- Go to [Anaconda installation page](https://www.anaconda.com/products/individual), and download the up-to-date version (pkg) for Mac (see [this link](https://towardsdatascience.com/install-anaconda-on-macos-big-sur-9fbd7c4b6c24) for the following steps). 
- Once it is installed, you should see
  ![img](https://miro.medium.com/max/1400/1*0XIG0OOzT6_0eyVrDurXMQ.png)



- Check the directory of Anaconda-Navigator, usually it is in /Users/dongzhang/opt/anaconda3/. Do activation by

  ```
  source /Users/dongzhang/opt/anaconda3/bin/activate
  ```

- Next step (current Mac has zsh shell):

  ```
  conda init zsh
  ```

- Now you have (base) in front of your terminal like this:

  ```
  (base) dongzhang@Dongs-MBP-2
  ```

  And your conda is working. 

- To remove the (base) when conda is still active, you can do

  ```
  conda config --set changeps1 false
  ```

- Note that for active and deactivate (see [this link](https://apple.stackexchange.com/questions/371727/how-do-i-remove-the-source-base-from-my-terminal) for more details):

  ```
  # run this in an interactive shell
  # enable environment called "base", the default env from conda
  conda activate base
  # deactivate an environment
  conda deactivate
  ```



- To deactivate conda by default, you can do

  ```
  conda config --set auto_activate_base false
  ```

  See [this link](http://progsharing.blogspot.com/2019/07/deactivate-condas-base-environment-on.html).

  To active conda just use

  ```
  conda activate
  ```




### Facebook Prophet

You may have issue if you directly install:

```
pip3 install fbprophet
```

and

```python
from fbprophet import Prophet
```

Do the following instead:

```
pip3 uninstall fbprophet
pip3 install pystan==2.19.1.1
pip3 install prophet
```

Then do

```python
from prophet import Prophet
```

[reference](https://stackoverflow.com/questions/67820844/stanmodel-object-has-no-attribute-fit-class-while-using-prophet)

