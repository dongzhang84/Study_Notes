# GitHub Notes



## Add Exisiting Project to Git

First go to the working directory which you would like to sync with GitHub, initalize the local directory as a Git repository:

```
git init
```

then add the fields in your new local repository by

```
git add .
```

Commit the files that you have staged in your local repository:

```
git commmit -m "First commit"
```

Next, add the URL for the remote repository where your local repostory will be pushed:

```
git remote add origin <remote repository URL>
```

The step of pushing files (changes) in your local reposityr to GitHub by:

```
git push origin main
```

or

```
git push origin master
```

If they does not work, we can use:

```
git push origin main --force
```

or

```
git push origin master --force
```





**Remove .DS_Store files**

To remove .DS_Store fiels from a Git repository, you can first remove exisiting fields from the repository: 

```
find . -name .DS_Store -print0 | xargs -0 git rm -f --ignore-unmatch
```

then do 

```
echo .DS_Store >> .gitignore
```

More details see [this link](https://stackoverflow.com/questions/107701/how-can-i-remove-ds-store-files-from-a-git-repository).

### Update Github Repo

```
git add .
```

Then do commit:

```
git commmit -m "updated"
```

Check if the repo is connected to the directory:

```
git remote -v
```

Then do the push:

```
git push origin main --force
```



### Personal  Access Token

Note that since August 13 2021 you need to use a personal access token instead of passowrd authentication. 

You can follow [this link](https://blog.csdn.net/weixin_41010198/article/details/119698015), or github account, **Settings** --> **Developer setting** --> **Personal access tokens** -- > Select "repo" and "delete repo" --> **Generate new token**

Use your email address and token as password to access to github. 

Add in your MacOS, you can do (see more details [here](https://whatibroke.com/2021/08/14/support-for-password-authentication-was-removed-on-august-13-2021-please-use-a-personal-access-token-instead-fix-for-mac/)):

1. go to KeyChain Access,
2. Search for "GitHub",
3. then when then result "github.com" pops up, change the account or password to your new account, and save.

