====================
Demos for Q_learning
====================

------------------
How to prepare
------------------

- pip
Just run ``pip install -r requirements.txt`` to install the dependencies. Be careful with the Python version and global packages.

- virtualenv
If you're familiar with ``virtualenv``, then you can create the environment by

.. code-block:: shell

    virtualenv demo

and activate the virtual environment

.. code-block:: shell

    source bin/activate

Finally, use ``pip`` to install the requirements

.. code-block:: shell

    pip install -r requirements.txt

Of course, ``virtualenvwrapper`` is more pleasant.

- pipenv(highly recommended)
If you can use ``pipenv``, that's perfect.
Use ``pipenv`` to create a working directory

.. code-block:: shell

    pipenv --python 3.6

(Python 3.6 is great)
and run

.. code-block:: shell

    pipenv install

to install all the dependencies for this project

------------------
How to run
------------------

- pip/virtualenv

Run ``python3.6 main.py -h`` directly to see the help page.

- pipenv

Run ``pipenv run python3.6 main.py -h`` to see the help.

-----------
USAGE
-----------

::

    usage: main.py [-h] {train,run} ...

    This is a demo to show how Q_learning makes agent intelligent

    optional arguments:
      -h, --help   show this help message and exit

    mode:
      {train,run}  Choose a mode
        train      Train an agent
        run        Make an agent run

*************
train
*************

Help for train subcommand

:: 

    usage: main.py train [-h] [-m {c,r}] [-r ROUND] [-l] [-s] [-c CONFIG_FILE]
    			[-d {t}] [-a]

    optional arguments:
      -h, --help            show this help message and exit
      -m {c,r}, --mode {c,r}
                            Training mode, by rounds or by convergence
      -r ROUND, --round ROUND
                            Training rounds, neglect when convergence is chosen
      -l, --load            Whether to load Q table from a csv file when training
      -s, --show            Show the training process.
      -c CONFIG_FILE, --config_file CONFIG_FILE
                            Config file for significant parameters
      -d {t}, --demo {t}    Choose a demo to run
      -a, --heuristic       Whether to use a heuristic iteration

Details:

- m

Mode of terminal when training. ``c`` stands for 'convergence', ``r`` stands for 'round'. If ``c`` is chosen, then the agent will stop only when the Q table is converged.  If ``r`` is chosen, the agent will only be trained for certain rounds(which can be modified by ``-r`` flag).

- l

Load the Q table from a csv file. The file name can be modified in the program. If not, a new Q table is built.

- r

Number of rounds to train the warrior. Will be ignored is ``-m c`` is chosen.

- s

``s`` flag can show the process of training if been selected.

- c

A config filename can be specified when training with this argument.

- d

Choose a demo to train.

- a

Whether to use the heuristic policy to accelerate the training progress.


*************
run
*************

Help for run subcommand

::

    usage: main.py run [-h] [-d {t}] [-q Q]

    optional arguments:
      -h, --help          show this help message and exit
      -d {t}, --demo {t}  Choose a demo to run
      -q Q                Choose a Q table from a csv file

Details:

- d

Choose a demo to run.

- q

Specify a Q table file to use when run.

-------------
Demos
-------------

****************
1-D TreasureHunt
****************

################
Config file
################

Config file must be a YAML file containing the following parameters

.. code-block:: yaml

  size: 10
  epsilon: 0.9
  gamma: 0.9
  alpha: 0.1
  speed: 0.1


- size

The length of the map.

- epsilon

The probability of choosing a random action. The other option is choosing the action which makes the Q value of current state maximum.

- gamma

Discount factor.

- alpha

Learning rate.


- speed

Speed of displaying.

###################
DISPLAY
###################

After convegence of training::

   Xo_________T
   X_o________T
   X__o_______T
   X___o______T
   X____o_____T
   X_____o____T
   X______o___T
   X_______o__T
   X________o_T
   X_________oT
   X__________o

The agent can find the treasure directly.

*******************
2-D TreasureHunt
*******************

###################
DISPLAY
###################

::

|@| | |+| | | | | | |
| |+|X| | | | |+| | |
| | |X| | | | | | | |
| | | | | |X|X|+| | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | |X| | |+| |
| | | | | | |X|X| |+|
| | |+| | | | | | | |
| | | | |+| | | |X|#|

-------------------
Thanks
-------------------

`莫烦PYTHON <https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/2-1-general-rl/>`_
