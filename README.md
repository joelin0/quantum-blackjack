# quantum-blackjack
Utilities for Quantum Blackjack and Quantum Strategies in Communication Limited Games

# Requirements
Uses Python 2.7.

Requires [numpy](http://www.numpy.org/) and [PICOS](http://picos.zib.de/). These can be downloaded using ```pip```.

Quick installation can also be performed by running
```
pip install -r requirements.txt
```

# File Organization
`blackjack.py` contains code for calculating expected payouts in our modified version of blackjack. It can also be used to find game configurations leading to quantum advantage. It relies on `strategies.py`,
which more generally provides the infrastructure for calculating optimal strategies in communication limited games. Finally, `hyperbit_algorithm.py` is an example file that shows how `blackjack.py` can be
used to determine the optimal hyperbit strategy of a blackjack configuration and determine the rotation angles of the corresponding quantum circuit.

`advantageous_configurations` contains several text files listing configurations found to have quantum advantage using hyperbit strategies.

# Paper
The code in this repository was used for the paper _Quantum Blackjack or Can MIT Bring Down the House Again?_ by Joseph Lin, Joseph Formaggio, Aram Harrow, and Anand Natarajan.
