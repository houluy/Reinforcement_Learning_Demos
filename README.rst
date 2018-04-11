===================
Demo for Q_learning
===================


------------------
How to prepare
------------------

- pip
Just run ``pip install -r requirements.txt`` to install the dependencies. Be careful with the Python version and global packages.

- virtualenv
If you're familiar with ``virtualenv``, then you can create the environment by::

    virtualenv demo

and activate the virtual environment::

    source bin/activate

Finally, use ``pip`` to install the requirements::

    pip install -r requirements.txt

Of course, ``virtualenvwrapper`` is more pleasant.

- pipenv(highly recommended)
If you can use ``pipenv``, that's perfect.
Use ``pipenv`` to create a working directory::

    pipenv --python 3.6

(Python 3.6 is great)
and run::

    pipenv install

to install all the dependencies for this project

------------------
How to run
------------------

- pip/virtualenv

Run ``python3.6 find_treasure.py -h`` directly to see the help page.

- pipenv

Run ``pipenv run python3.6 find_treasure.py -h`` to see the help.

**********
USAGE
**********

::

    usage: find_treasure.py [-h] [-l] [-r ROUNDS] [-m {t,p}] [-s] [-c CONFIG_FILE]

    This is a demo to show how Q_learning makes agent intelligent

    optional arguments:
        -h, --help          show this help message and exit
        -l, --load          Load Q table from a csv file
        -r ROUNDS, --rounds ROUNDS
                            Training rounds
        -m {t,p}, --mode {t,p}
                            Mode: oneof ["t"(train), "p"(play)]
        -s, --show          Show the training process.
        -c CONFIG_FILE, --config_file CONFIG_FILE
                            Config file for significant parameters

- -l

Load the Q table from a csv file. The file name can be modified in the program.

- -r

Number of rounds to train the warrior. 


************
CONFIG
************

Config file must be a YAML file containing the following parameters::

  size: 10
  epsilon: 0.9
  gamma: 0.9
  alpha: 0.1
  instant_reward: 1
  speed: 0.1

- size

The length of the map.

- epsilon

The probability of choosing a random action. The other option is choosing the action which makes the Q value of current state maximum

- gamma

The decay rate of the expectation of feature.

- alpha


-------------------
THANKS
-------------------

`莫烦PYTHON: <https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/2-1-general-rl/>`_
