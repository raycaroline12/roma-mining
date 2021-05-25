# Roma Mining

## Requirements

* Python 3.7.6+ 64-bit
* Python Virtual Environment

All commands for this repo must be executed with a virtual environment activated to keep dependencies isolated and better deal with Django 3 commands.

Review on this using virtualenv:

1. Creating a virtual environment named "myenv":

```sh
python -m venv myenv
```
2. Activating and deactivating virtual environments:
* Windows

```sh
myenv\scripts\activate
...

(do what you need to do)

deactivate
```
* Linux

```sh
source myenv/bin/activate
...

(do what you need to do)

deactivate
```

## Instalation

1. Cloning the repository

```sh
git clone https://github.com/raycaroline12/roma-mining.git
```

2. Installing dependencies

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Basic Usage

* Running the server:

To run the server, execute the following command:

```sh
python manage.py runserver [port]
```
That will provide you a development server. You just put the address provided on your browser. Something like, http://127.0.0.1:8000/, if you've placed 8000 as [port].


