# StreamlitアプリをHerokuでティプロイする

## 1. 3つのファイルを作成

### `Procfile`

`main.py`の部分はstreamlitのエントリーポイント

```Procfile
web: sh setup.sh && streamlit run main.py
```

### `requirements.txt`

```requirements.txt
streamlit
--find-links https://download.pytorch.org/whl/torch_stable.html
torch==1.7.1+cpu
torchvision
```

### `setup.sh`

```setup.sh
mkdir -p ~/.streamlit

echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

## 2. Heroku設定

tips
```
$ heroku plugins:install heroku-accounts
$ heroku accounts:add account1
$ heroku accounts:set account2
$ heroku accounts:remove account2
$ heroku accounts
```

## 3. Deploy

1. `heroku create name`
2. `git push heroku master` # with valid commit
